#!/usr/bin/env python3
"""Stdlib-only training run discovery, log parser, and standalone HTTP server."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import glob
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


HEAD_BYTES = 32 * 1024
TAIL_BYTES = 4 * 1024 * 1024
LIVE_THRESHOLD_S = 300.0
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "external",
    ".pytest_cache",
    "build",
    "dist",
    ".mypy_cache",
}
EVAL_PAIR_RE = re.compile(r"([A-Za-z0-9_.-]+)=([+-]?[0-9]+(?:\.[0-9]+)?)%")

# Orbit Wars League PPO
OW_PPO_HDR_RE = re.compile(
    r"\[RL-League-PPO\]\s+Loaded\s+(\S+)(?:.*?\|\s*dev=(\S+))?(?:.*?\|\s*iters=(\d+))?"
    r"(?:.*?workers=(\d+))?(?:.*?lr=([0-9eE.+-]+))?"
)
OW_PPO_BASE_RE = re.compile(r"\[Baseline EVAL\((\d+)g\)\]\s+(.+)")
OW_PPO_ITER_RE = re.compile(
    r"^\s*iter\s+(\d+)/(\d+)\s+\(\s*([0-9.]+)s"
    r"(?:\s+\[roll=([0-9.]+)s\s+upd=([0-9.]+)s\])?\)\s*\|\s*"
    r"states=(\d+)\s*\|\s*2P=\s*([0-9.]+)%"
    r"(?:\s*\(sw=(\d+),\s*sp=(\d+)\))?"
    r"(?:\s+4P=\s*([0-9.]+)%\(rk=([0-9.]+)\))?\s*\|\s*"
    r"rew=([+-]?[0-9.]+)\s+kl=([0-9.]+)\s+akl=([0-9.]+)\s+ent=([0-9.]+)\s+vl=([0-9.]+)"
    r"(?:\s+aux=([0-9.]+)%)?"
    r"(?:\s*\|\s*EVAL\((\d+)g\):\s*(.+))?"
)
OW_PPO_DONE_RE = re.compile(r"\[RL-League-PPO\]\s+Saved best ->")

# Orbit Wars Soft-KD / 4M Ensemble
SKD_ENS_RE = re.compile(r"\[Ensemble4M\]\s+Device=(\S+)")
SKD_PHASE_RE = re.compile(r"\[(Ensemble4M|SoftKD)\s+(Collect|Label)\]\s+(.+)")
SKD_HDR_RE = re.compile(
    r"\[SoftKD Train\]\s+(.+?)\s*->\s*([0-9.]+[MK])\s+params\s+\(device=([^)]+)\)\s*\|"
    r".*?epochs=(\d+)(?:.*?batch=(\d+))?(?:.*?lr=([0-9eE.+-]+))?"
)
SKD_EP_RE = re.compile(
    r"^\s*ep\s+(\d+)/(\d+)\s+\(\s*([0-9.]+)s,\s*([0-9.]+)s/ep\)\s*\|\s*"
    r"tr_kl=([0-9.]+)\s+val_kl=([0-9.]+)\s*\|\s*"
    r"val_all=\s*([0-9.]+)%\s+val_launch=\s*([0-9.]+)%\s+val_tgt=\s*([0-9.]+)%\s+"
    r"aux_own=\s*([0-9.]+)%\s+vsign=\s*([0-9.]+)%"
)
SKD_EVAL_RE = re.compile(r"\[Ensemble4M Eval\((\d+)g\)\]\s+(.+)")
SKD_DONE_RE = re.compile(r"\[SoftKD Train\]\s+Saved best")

# Galcon PPO / Supervised
GALCON_PPO_HDR_RE = re.compile(
    r"\[(Galcon[^\]]*PPO[^\]]*)\]\s+Start\s+(\d+)g[^=]*=\s*([0-9.]+)%(?:\s*\(pool=(\d+)\))?"
)
GALCON_PPO_ITER_RE = re.compile(
    r"\[Galcon[^\]]*PPO\]\s+iter\s+(\d+)/(\d+)\s+\(eps=(\d+),\s*([0-9.]+)s"
    r"(?:\s+\[roll=([0-9.]+)s\s+upd=([0-9.]+)s\])?\)\s*\|\s*"
    r"wins=(\d+)/(\d+)\s*\(([^)]*)\)\s*\|\s*states=(\d+)\s*\|\s*pool=(\d+)"
    r"(?:\s*\|\s*val_\d+g=([0-9.]+)%(.*))?"
)
GALCON_RATIO_RE = re.compile(r"(sp|bots|boss)=(\d+)/(\d+)")
GALCON_SUP_EP_RE = re.compile(
    r"^\s*\[([^\]]+)\]\s+ep\s+(\d+)/(\d+)\s+\(\s*([0-9.]+)s\)\s*\|\s*"
    r"val_bce=([0-9.]+)\s+rec=\s*([0-9.]+)%\s+prec=\s*([0-9.]+)%\s+"
    r"aux=\s*([0-9.]+)%\s+vsign=\s*([0-9.]+)%"
)
GALCON_DONE_RE = re.compile(r"\[Galcon[^\]]*(?:Promoted|Saved)\b|Held-Out Multi-Map Gauntlet")

# Generic fallback log line
GEN_STEP_RE = re.compile(
    r"(?:^|\b)(?:epoch|ep|iter|iteration|step)\s*[:=#]?\s*(\d+)(?:\s*/\s*(\d+))?",
    re.IGNORECASE,
)
GEN_KV_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_./-]{0,24})\s*=\s*([+-]?\d+(?:\.\d+)?)(%?)")


def _read_text_bounded(path: Path) -> str:
    """Read small files completely or head+tail slices for large logs."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size <= HEAD_BYTES + TAIL_BYTES:
            return fh.read().decode("utf-8", errors="replace")
        head = fh.read(HEAD_BYTES)
        fh.seek(max(HEAD_BYTES, size - TAIL_BYTES))
        tail = fh.read(TAIL_BYTES)
    nl = tail.find(b"\n")
    if nl != -1:
        tail = tail[nl + 1 :]
    return head.decode("utf-8", errors="replace") + "\n...\n" + tail.decode("utf-8", errors="replace")


def _parse_pct_pairs(fragment: str) -> OrderedDict[str, float]:
    out: OrderedDict[str, float] = OrderedDict()
    for key, val in EVAL_PAIR_RE.findall(fragment):
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def _clean_workspace_paths(raw_list: List[str]) -> List[Path]:
    seen = set()
    roots: List[Path] = []
    for item in raw_list:
        for part in item.split(","):
            s = part.strip()
            if not s:
                continue
            if s.startswith("file://"):
                s = unquote(urlparse(s).path)
            p = Path(s).expanduser()
            try:
                rp = p.resolve()
            except OSError:
                continue
            if rp.is_dir() and str(rp) not in seen:
                seen.add(str(rp))
                roots.append(rp)
    return roots


def _discover_sources(app_data: Path, workspace_paths: List[Path], max_age_days: int) -> List[Tuple[Path, str]]:
    """Return candidate (path, source_label) pairs within max_age_days."""
    cutoff = time.time() - max(1, max_age_days) * 86400.0
    found: Dict[str, Tuple[Path, str]] = {}

    brain_roots: List[Path] = []
    primary_brain = app_data / "brain"
    if primary_brain.is_dir():
        brain_roots.append(primary_brain)
    else:
        for cand in glob.glob(str(Path.home() / ".gemini" / "*" / "brain")):
            p = Path(cand)
            if p.is_dir():
                brain_roots.append(p)

    for broot in brain_roots:
        for pat in ("*/.system_generated/tasks/*.log", "*/scratch/*metrics*.jsonl"):
            for m in glob.glob(str(broot / pat)):
                p = Path(m)
                try:
                    st = p.stat()
                except OSError:
                    continue
                if st.st_mtime >= cutoff and st.st_size > 0:
                    conv_id = p.parts[len(broot.parts)] if len(p.parts) > len(broot.parts) else ""
                    label = f"{p.name}" + (f" ({conv_id[:8]})" if conv_id else "")
                    found[str(p.resolve())] = (p, label)

    jsonl_names = {"metrics.jsonl", "train_metrics.jsonl"}
    ws_count = 0
    for wroot in workspace_paths:
        for cur, dirs, files in os.walk(wroot):
            rel_depth = len(Path(cur).relative_to(wroot).parts) if cur != str(wroot) else 0
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            if rel_depth >= 3:
                dirs[:] = []
            for fname in files:
                if fname in jsonl_names or fname.endswith(".metrics.jsonl"):
                    p = Path(cur) / fname
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_mtime >= cutoff and st.st_size > 0:
                        rel = str(p.relative_to(wroot))
                        found[str(p.resolve())] = (p, f"{wroot.name}/{rel}")
                        ws_count += 1
                        if ws_count >= 50:
                            break
            if ws_count >= 50:
                break
    return list(found.values())


def _median_step_time(elapsed_points: List[Tuple[int, float]]) -> Optional[float]:
    """Compute seconds-per-step using median of recent consecutive elapsed deltas."""
    deltas: List[float] = []
    for i in range(1, len(elapsed_points)):
        dx = elapsed_points[i][0] - elapsed_points[i - 1][0]
        dt = elapsed_points[i][1] - elapsed_points[i - 1][1]
        if dx > 0 and dt > 0:
            deltas.append(dt / dx)
    if deltas:
        return statistics.median(deltas[-5:])
    if elapsed_points and elapsed_points[-1][0] > 0 and elapsed_points[-1][1] > 0:
        return elapsed_points[-1][1] / elapsed_points[-1][0]
    return None


def _compute_eta_s(
    status: str,
    step: int,
    total_steps: Optional[int],
    elapsed_points: List[Tuple[int, float]],
    age_s: float,
) -> Tuple[Optional[float], Optional[float]]:
    sec_per_step = _median_step_time(elapsed_points)
    if status != "live" or total_steps is None or step <= 0 or step >= total_steps or sec_per_step is None:
        return sec_per_step, None
    rem = (total_steps - step) * sec_per_step - min(age_s, sec_per_step * 0.85)
    return sec_per_step, max(0.0, rem)


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val:.1f}%" if val is not None else "—"


def _fmt_num(val: Optional[float], digits: int = 3) -> str:
    return f"{val:.{digits}f}" if val is not None else "—"


def _make_series(key: str, label: str, points: List[Dict[str, Any]], bold: bool = False) -> Dict[str, Any]:
    return {"key": key, "label": label, "bold": bold, "points": points}


def _parse_orbit_ppo(text: str, path: Path, label: str, mtime: float) -> Optional[Dict[str, Any]]:
    hdr_m = OW_PPO_HDR_RE.search(text)
    base_m = OW_PPO_BASE_RE.search(text)
    lines = text.splitlines()
    rows: List[Dict[str, Any]] = []
    for ln in lines:
        m = OW_PPO_ITER_RE.match(ln)
        if m:
            evals = _parse_pct_pairs(m.group(19) or "")
            is_best = "*BEST*" in (m.group(19) or "")
            rows.append(
                {
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)),
                    "elapsed_s": float(m.group(3)),
                    "roll_s": float(m.group(4)) if m.group(4) else None,
                    "upd_s": float(m.group(5)) if m.group(5) else None,
                    "states": int(m.group(6)),
                    "rollout_wr": float(m.group(7)),
                    "wr_4p": float(m.group(10)) if m.group(10) else None,
                    "rk_4p": float(m.group(11)) if m.group(11) else None,
                    "rew": float(m.group(12)),
                    "kl": float(m.group(13)),
                    "akl": float(m.group(14)),
                    "ent": float(m.group(15)),
                    "vl": float(m.group(16)),
                    "aux": float(m.group(17)) if m.group(17) else None,
                    "eval_games": int(m.group(18)) if m.group(18) else None,
                    "evals": evals,
                    "is_best": is_best,
                }
            )

    if not hdr_m and not base_m and not rows:
        return None

    ckpt = Path(hdr_m.group(1)).stem if hdr_m else "orbit_ppo"
    device = hdr_m.group(2) if hdr_m and hdr_m.group(2) else None
    total_steps = int(hdr_m.group(3)) if hdr_m and hdr_m.group(3) else (rows[-1]["total_steps"] if rows else None)
    workers = hdr_m.group(4) if hdr_m else None
    lr = hdr_m.group(5) if hdr_m else None

    base_evals = _parse_pct_pairs(base_m.group(2)) if base_m else OrderedDict()
    cur_step = rows[-1]["step"] if rows else 0
    is_done = bool(OW_PPO_DONE_RE.search(text)) or (total_steps is not None and cur_step >= total_steps and cur_step > 0)

    # Collect eval point series (step 0 baseline + eval iterations)
    eval_keys: List[str] = []
    for k in base_evals:
        if k not in eval_keys:
            eval_keys.append(k)
    for r in rows:
        for k in r["evals"]:
            if k not in eval_keys:
                eval_keys.append(k)
    if "avg" in eval_keys:
        eval_keys = ["avg"] + [k for k in eval_keys if k != "avg"]

    primary_series: List[Dict[str, Any]] = []
    for k in eval_keys:
        pts: List[Dict[str, Any]] = []
        if k in base_evals:
            pts.append({"x": 0, "y": base_evals[k], "best": False})
        for r in rows:
            if k in r["evals"]:
                pts.append({"x": r["step"], "y": r["evals"][k], "best": r["is_best"] if k == "avg" else False})
        if pts:
            lbl = "Eval Avg" if k == "avg" else k
            primary_series.append(_make_series(k, lbl, pts, bold=(k == "avg")))

    # Best and latest eval stats
    best_eval: Optional[Tuple[int, float]] = None
    latest_eval: Optional[Tuple[int, float]] = None
    if "avg" in base_evals:
        best_eval = (0, base_evals["avg"])
        latest_eval = (0, base_evals["avg"])
    for r in rows:
        if "avg" in r["evals"]:
            val = r["evals"]["avg"]
            latest_eval = (r["step"], val)
            if best_eval is None or val >= best_eval[1]:
                best_eval = (r["step"], val)

    secondary = [
        {
            "id": "kl",
            "label": "KL & Anchor",
            "title": "KL & Anchor Divergence",
            "unit": "",
            "series": [
                _make_series("kl", "KL", [{"x": r["step"], "y": r["kl"]} for r in rows], bold=True),
                _make_series("akl", "Anchor KL", [{"x": r["step"], "y": r["akl"]} for r in rows]),
            ],
        },
        {
            "id": "vl",
            "label": "Value Loss",
            "title": "Critic Value Loss",
            "unit": "",
            "series": [_make_series("vl", "Value Loss", [{"x": r["step"], "y": r["vl"]} for r in rows], bold=True)],
        },
        {
            "id": "ent",
            "label": "Entropy",
            "title": "Policy Entropy",
            "unit": "",
            "series": [_make_series("ent", "Entropy", [{"x": r["step"], "y": r["ent"]} for r in rows], bold=True)],
        },
        {
            "id": "rollout",
            "label": "2P Rollout %",
            "title": "Self-Play & League Rollout Win Rate (%)",
            "unit": "%",
            "y_min": 0,
            "y_max": 100,
            "ref_line": 50.0,
            "ref_label": "50%",
            "series": [_make_series("rollout_wr", "2P Win %", [{"x": r["step"], "y": r["rollout_wr"]} for r in rows], bold=True)],
        },
    ]
    if any(r["aux"] is not None for r in rows):
        secondary.append(
            {
                "id": "aux",
                "label": "Aux t+10 %",
                "title": "Future Ownership Aux Accuracy (%)",
                "unit": "%",
                "y_min": 50,
                "y_max": 100,
                "series": [
                    _make_series(
                        "aux",
                        "Aux Own %",
                        [{"x": r["step"], "y": r["aux"]} for r in rows if r["aux"] is not None],
                        bold=True,
                    )
                ],
            }
        )
    if any(r["roll_s"] is not None for r in rows):
        secondary.append(
            {
                "id": "timing",
                "label": "Step Time",
                "title": "Iteration Time Breakdown (s)",
                "unit": "s",
                "series": [
                    _make_series(
                        "roll_s",
                        "Rollout (s)",
                        [{"x": r["step"], "y": r["roll_s"]} for r in rows if r["roll_s"] is not None],
                        bold=True,
                    ),
                    _make_series(
                        "upd_s",
                        "Update (s)",
                        [{"x": r["step"], "y": r["upd_s"]} for r in rows if r["upd_s"] is not None],
                    ),
                ],
            }
        )

    last = rows[-1] if rows else None
    kpis = [
        {
            "label": "Best Eval Avg",
            "value": _fmt_pct(best_eval[1]) if best_eval else "—",
            "sub": (
                f"@ iter {best_eval[0]}" + (f" (base {base_evals['avg']:.1f}%)" if "avg" in base_evals and best_eval[0] > 0 else "")
                if best_eval
                else "awaiting first eval"
            ),
            "tone": "good" if best_eval and "avg" in base_evals and best_eval[1] > base_evals["avg"] else "neutral",
        },
        {
            "label": "Latest Eval Avg" if latest_eval else "2P Rollout WR",
            "value": _fmt_pct(latest_eval[1]) if latest_eval else _fmt_pct(last["rollout_wr"] if last else None),
            "sub": (
                f"@ iter {latest_eval[0]} · 2P {_fmt_pct(last['rollout_wr'] if last else None)}"
                if latest_eval and last
                else (f"rew {last['rew']:+.2f}" if last else "running baseline eval")
            ),
            "tone": "neutral",
        },
        {
            "label": "Aux Own t+10",
            "value": _fmt_pct(last["aux"] if last else None),
            "sub": f"states {last['states']:,}" if last else "—",
            "tone": "neutral",
        },
        {
            "label": "Policy KL / VL",
            "value": _fmt_num(last["kl"], 4) if last else "—",
            "sub": f"ent {last['ent']:.3f} · vl {last['vl']:.3f}" if last else "—",
            "tone": "neutral",
        },
    ]

    table_cols = ["Iter", "Eval Avg", "2P WR", "KL", "Ent", "VL", "Aux %"]
    table_rows = []
    for r in reversed(rows[-15:]):
        ev_avg = f"{r['evals']['avg']:.1f}%{'*' if r['is_best'] else ''}" if "avg" in r["evals"] else "—"
        table_rows.append(
            [
                f"{r['step']}/{r['total_steps']}",
                ev_avg,
                f"{r['rollout_wr']:.1f}%",
                f"{r['kl']:.4f}",
                f"{r['ent']:.3f}",
                f"{r['vl']:.3f}",
                f"{r['aux']:.1f}%" if r["aux"] is not None else "—",
            ]
        )

    headline_parts = []
    if best_eval:
        headline_parts.append(f"best {best_eval[1]:.1f}%")
    if last and last["aux"] is not None:
        headline_parts.append(f"aux {last['aux']:.1f}%")
    elif last:
        headline_parts.append(f"2P {last['rollout_wr']:.1f}%")

    return {
        "kind": "orbit_ppo",
        "kind_label": "Orbit Wars PPO",
        "title": f"Orbit Wars PPO · {ckpt}",
        "step": cur_step,
        "total_steps": total_steps,
        "step_label": "iter",
        "elapsed_s": last["elapsed_s"] if last else None,
        "elapsed_points": [(r["step"], r["elapsed_s"]) for r in rows],
        "is_done": is_done,
        "phase_note": "Running baseline eval..." if not rows and hdr_m else None,
        "meta": {k: v for k, v in [("device", device), ("workers", workers), ("lr", lr)] if v},
        "timing": _avg_timing(rows),
        "kpis": kpis,
        "charts": {
            "primary": {
                "title": "Eval Win Rate (%)",
                "unit": "%",
                "x_label": "iter",
                "y_min": 0,
                "y_max": 100,
                "ref_line": 50.0,
                "ref_label": "50% even",
                "series": primary_series,
            },
            "secondary": secondary,
        },
        "table": {"columns": table_cols, "rows": table_rows},
        "headline": " · ".join(headline_parts) if headline_parts else "baseline eval",
    }


def _parse_orbit_softkd(text: str, path: Path, label: str, mtime: float) -> Optional[Dict[str, Any]]:
    ens_m = SKD_ENS_RE.search(text)
    hdr_m = SKD_HDR_RE.search(text)
    eval_m = SKD_EVAL_RE.search(text)
    rows: List[Dict[str, Any]] = []
    phase_notes: List[str] = []

    for ln in text.splitlines():
        m = SKD_EP_RE.match(ln)
        if m:
            rows.append(
                {
                    "step": int(m.group(1)),
                    "total_steps": int(m.group(2)),
                    "elapsed_s": float(m.group(3)),
                    "step_s": float(m.group(4)),
                    "tr_kl": float(m.group(5)),
                    "val_kl": float(m.group(6)),
                    "val_all": float(m.group(7)),
                    "val_launch": float(m.group(8)),
                    "val_tgt": float(m.group(9)),
                    "aux_own": float(m.group(10)),
                    "vsign": float(m.group(11)),
                }
            )
            continue
        pm = SKD_PHASE_RE.search(ln)
        if pm:
            phase_notes.append(pm.group(0).strip())

    if not hdr_m and not ens_m and not rows:
        return None

    params = hdr_m.group(2) if hdr_m else None
    device = (hdr_m.group(3) if hdr_m else None) or (ens_m.group(1) if ens_m else None)
    total_steps = int(hdr_m.group(4)) if hdr_m else (rows[-1]["total_steps"] if rows else None)
    batch = hdr_m.group(5) if hdr_m else None
    lr = hdr_m.group(6) if hdr_m else None
    cur_step = rows[-1]["step"] if rows else 0
    is_done = bool(SKD_DONE_RE.search(text) or eval_m) or (
        total_steps is not None and cur_step >= total_steps and cur_step > 0
    )

    post_eval = _parse_pct_pairs(eval_m.group(2)) if eval_m else OrderedDict()
    best_row = min(rows, key=lambda r: r["val_kl"]) if rows else None
    last = rows[-1] if rows else None

    primary = {
        "title": "Distillation KL Divergence (lower is better)",
        "unit": "",
        "x_label": "ep",
        "series": [
            _make_series(
                "val_kl",
                "Val KL",
                [{"x": r["step"], "y": r["val_kl"], "best": bool(best_row and r["step"] == best_row["step"])} for r in rows],
                bold=True,
            ),
            _make_series("tr_kl", "Train KL", [{"x": r["step"], "y": r["tr_kl"]} for r in rows]),
        ],
    }

    secondary = [
        {
            "id": "accs",
            "label": "Val Acc %",
            "title": "Validation Policy & Aux Accuracies (%)",
            "unit": "%",
            "y_min": 0,
            "y_max": 100,
            "series": [
                _make_series("val_all", "val_all", [{"x": r["step"], "y": r["val_all"]} for r in rows], bold=True),
                _make_series("val_tgt", "val_tgt", [{"x": r["step"], "y": r["val_tgt"]} for r in rows]),
                _make_series("val_launch", "val_launch", [{"x": r["step"], "y": r["val_launch"]} for r in rows]),
                _make_series("aux_own", "aux_own", [{"x": r["step"], "y": r["aux_own"]} for r in rows]),
                _make_series("vsign", "vsign", [{"x": r["step"], "y": r["vsign"]} for r in rows]),
            ],
        },
        {
            "id": "time",
            "label": "Epoch Time",
            "title": "Seconds per Epoch (s)",
            "unit": "s",
            "series": [_make_series("step_s", "s/epoch", [{"x": r["step"], "y": r["step_s"]} for r in rows], bold=True)],
        },
    ]

    kpis = [
        {
            "label": "Best Val KL",
            "value": _fmt_num(best_row["val_kl"], 3) if best_row else "—",
            "sub": f"@ ep {best_row['step']} · tr {best_row['tr_kl']:.3f}" if best_row else "collecting/labeling",
            "tone": "good" if best_row else "neutral",
        },
        {
            "label": "Val All / Launch",
            "value": _fmt_pct(last["val_all"] if last else None),
            "sub": f"launch {_fmt_pct(last['val_launch'] if last else None)} · tgt {_fmt_pct(last['val_tgt'] if last else None)}",
            "tone": "neutral",
        },
        {
            "label": "Aux Own t+10",
            "value": _fmt_pct(last["aux_own"] if last else None),
            "sub": f"value sign {_fmt_pct(last['vsign'] if last else None)}",
            "tone": "neutral",
        },
        {
            "label": "Post-Eval Avg" if "avg" in post_eval else "Speed",
            "value": _fmt_pct(post_eval["avg"]) if "avg" in post_eval else (f"{last['step_s']:.1f}s/ep" if last else "—"),
            "sub": (
                " ".join(f"{k}={v:.1f}%" for k, v in post_eval.items() if k != "avg")[:44]
                if post_eval
                else (f"params {params}" if params else "Soft-KD")
            ),
            "tone": "neutral",
        },
    ]

    table_cols = ["Epoch", "Tr KL", "Val KL", "Val All", "Launch", "Tgt", "Aux Own"]
    table_rows = [
        [
            f"{r['step']}/{r['total_steps']}",
            f"{r['tr_kl']:.3f}",
            f"{r['val_kl']:.3f}",
            f"{r['val_all']:.1f}%",
            f"{r['val_launch']:.1f}%",
            f"{r['val_tgt']:.1f}%",
            f"{r['aux_own']:.1f}%",
        ]
        for r in reversed(rows[-15:])
    ]

    headline = (
        f"val_kl {best_row['val_kl']:.3f} · aux {last['aux_own']:.1f}%"
        if best_row and last
        else (phase_notes[-1][:48] if phase_notes else "initializing")
    )

    return {
        "kind": "orbit_softkd",
        "kind_label": "Orbit Wars Soft-KD",
        "title": f"Orbit Wars Soft-KD ({params})" if params else "Orbit Wars Soft-KD",
        "step": cur_step,
        "total_steps": total_steps,
        "step_label": "ep",
        "elapsed_s": last["elapsed_s"] if last else None,
        "elapsed_points": [(r["step"], r["elapsed_s"]) for r in rows],
        "is_done": is_done,
        "phase_note": phase_notes[-1] if not rows and phase_notes else None,
        "meta": {k: v for k, v in [("device", device), ("params", params), ("batch", batch), ("lr", lr)] if v},
        "timing": {"avg_step_s": statistics.mean(r["step_s"] for r in rows)} if rows else None,
        "kpis": kpis,
        "charts": {"primary": primary, "secondary": secondary},
        "table": {"columns": table_cols, "rows": table_rows},
        "headline": headline,
    }


def _parse_galcon_ppo(text: str, path: Path, label: str, mtime: float) -> Optional[Dict[str, Any]]:
    hdr_m = GALCON_PPO_HDR_RE.search(text)
    rows: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        m = GALCON_PPO_ITER_RE.search(ln)
        if not m:
            continue
        tot_w, tot_g = int(m.group(7)), max(1, int(m.group(8)))
        subs: Dict[str, float] = {}
        for sk, sw, sg in GALCON_RATIO_RE.findall(m.group(9) or ""):
            if int(sg) > 0:
                subs[f"{sk}_wr"] = round(100.0 * int(sw) / int(sg), 1)
        rows.append(
            {
                "step": int(m.group(1)),
                "total_steps": int(m.group(2)),
                "eps": int(m.group(3)),
                "elapsed_s": float(m.group(4)),
                "roll_s": float(m.group(5)) if m.group(5) else None,
                "upd_s": float(m.group(6)) if m.group(6) else None,
                "rollout_wr": round(100.0 * tot_w / tot_g, 1),
                "states": int(m.group(10)),
                "pool": int(m.group(11)),
                "val_wr": float(m.group(12)) if m.group(12) else None,
                "is_best": "*BEST*" in (m.group(13) or ""),
                **subs,
            }
        )

    if not hdr_m and not rows:
        return None

    base_val = float(hdr_m.group(3)) if hdr_m else None
    cur_step = rows[-1]["step"] if rows else 0
    total_steps = rows[-1]["total_steps"] if rows else None
    is_done = bool(GALCON_DONE_RE.search(text)) or (total_steps is not None and cur_step >= total_steps and cur_step > 0)

    val_pts: List[Dict[str, Any]] = []
    if base_val is not None:
        val_pts.append({"x": 0, "y": base_val, "best": False})
    for r in rows:
        if r["val_wr"] is not None:
            val_pts.append({"x": r["step"], "y": r["val_wr"], "best": r["is_best"]})

    primary_series = []
    if val_pts:
        primary_series.append(_make_series("val_wr", "Val Gauntlet %", val_pts, bold=True))
    primary_series.append(
        _make_series("rollout_wr", "Rollout %", [{"x": r["step"], "y": r["rollout_wr"]} for r in rows], bold=(not val_pts))
    )
    for k, lbl in (("boss_wr", "LuaBoss %"), ("bots_wr", "Bots %"), ("sp_wr", "Self-Play %")):
        pts = [{"x": r["step"], "y": r[k]} for r in rows if k in r]
        if pts:
            primary_series.append(_make_series(k, lbl, pts))

    secondary = [
        {
            "id": "states",
            "label": "States & Pool",
            "title": "Collected States per Iteration",
            "unit": "",
            "series": [_make_series("states", "States", [{"x": r["step"], "y": r["states"]} for r in rows], bold=True)],
        }
    ]
    if any(r["roll_s"] is not None for r in rows):
        secondary.insert(
            0,
            {
                "id": "timing",
                "label": "Step Time",
                "title": "Rollout vs Update Time (s)",
                "unit": "s",
                "series": [
                    _make_series("roll_s", "Rollout (s)", [{"x": r["step"], "y": r["roll_s"]} for r in rows if r["roll_s"] is not None], bold=True),
                    _make_series("upd_s", "Update (s)", [{"x": r["step"], "y": r["upd_s"]} for r in rows if r["upd_s"] is not None]),
                ],
            },
        )

    best_val = max(val_pts, key=lambda p: p["y"]) if val_pts else None
    last = rows[-1] if rows else None
    kpis = [
        {
            "label": "Best Val WR",
            "value": _fmt_pct(best_val["y"]) if best_val else "—",
            "sub": f"@ iter {best_val['x']}" + (f" (base {base_val:.1f}%)" if base_val is not None and best_val["x"] > 0 else "") if best_val else "—",
            "tone": "good" if best_val and base_val is not None and best_val["y"] > base_val else "neutral",
        },
        {
            "label": "Rollout Win %",
            "value": _fmt_pct(last["rollout_wr"] if last else None),
            "sub": f"boss {_fmt_pct(last.get('boss_wr')) if last else '—'} · pool {last['pool'] if last else '—'}",
            "tone": "neutral",
        },
        {
            "label": "Episodes / States",
            "value": f"{last['eps']:,}" if last else "—",
            "sub": f"states {last['states']:,}" if last else "—",
            "tone": "neutral",
        },
        {
            "label": "Roll / Upd",
            "value": f"{last['roll_s']:.1f}s / {last['upd_s']:.1f}s" if last and last["roll_s"] is not None else "—",
            "sub": f"total {last['elapsed_s']:.1f}s" if last else "—",
            "tone": "neutral",
        },
    ]

    return {
        "kind": "galcon_ppo",
        "kind_label": "Galcon PPO",
        "title": hdr_m.group(1) if hdr_m else "Galcon Gen4 PPO",
        "step": cur_step,
        "total_steps": total_steps,
        "step_label": "iter",
        "elapsed_s": last["elapsed_s"] if last else None,
        "elapsed_points": [(r["step"], r["elapsed_s"]) for r in rows],
        "is_done": is_done,
        "phase_note": None,
        "meta": {"pool": str(last["pool"])} if last else {},
        "timing": _avg_timing(rows),
        "kpis": kpis,
        "charts": {
            "primary": {
                "title": "Val & Rollout Win Rate (%)",
                "unit": "%",
                "x_label": "iter",
                "y_min": 0,
                "y_max": 100,
                "ref_line": 50.0,
                "ref_label": "50%",
                "series": primary_series,
            },
            "secondary": secondary,
        },
        "table": {
            "columns": ["Iter", "Val %", "Rollout %", "Boss %", "States", "Pool"],
            "rows": [
                [
                    f"{r['step']}/{r['total_steps']}",
                    f"{r['val_wr']:.1f}%{'*' if r['is_best'] else ''}" if r["val_wr"] is not None else "—",
                    f"{r['rollout_wr']:.1f}%",
                    _fmt_pct(r.get("boss_wr")),
                    f"{r['states']:,}",
                    str(r["pool"]),
                ]
                for r in reversed(rows[-15:])
            ],
        },
        "headline": f"val {best_val['y']:.1f}%" if best_val else "galcon ppo",
    }


def _parse_galcon_sup(text: str, path: Path, label: str, mtime: float) -> Optional[Dict[str, Any]]:
    rows = []
    title = None
    for ln in text.splitlines():
        m = GALCON_SUP_EP_RE.match(ln)
        if m:
            title = title or m.group(1)
            rows.append(
                {
                    "step": int(m.group(2)),
                    "total_steps": int(m.group(3)),
                    "elapsed_s": float(m.group(4)),
                    "val_bce": float(m.group(5)),
                    "rec": float(m.group(6)),
                    "prec": float(m.group(7)),
                    "aux": float(m.group(8)),
                    "vsign": float(m.group(9)),
                }
            )
    if not rows:
        return None
    best_r = min(rows, key=lambda r: r["val_bce"])
    last = rows[-1]
    return {
        "kind": "galcon_sup",
        "kind_label": "Galcon Supervised",
        "title": title or "Galcon Supervised",
        "step": last["step"],
        "total_steps": last["total_steps"],
        "step_label": "ep",
        "elapsed_s": last["elapsed_s"],
        "elapsed_points": [(r["step"], r["elapsed_s"]) for r in rows],
        "is_done": last["step"] >= last["total_steps"],
        "phase_note": None,
        "meta": {},
        "timing": None,
        "kpis": [
            {"label": "Best Val BCE", "value": f"{best_r['val_bce']:.3f}", "sub": f"@ ep {best_r['step']}", "tone": "good"},
            {"label": "Recall / Prec", "value": f"{last['rec']:.1f}% / {last['prec']:.1f}%", "sub": "multi-send", "tone": "neutral"},
            {"label": "Aux Own", "value": f"{last['aux']:.1f}%", "sub": f"vsign {last['vsign']:.1f}%", "tone": "neutral"},
            {"label": "Elapsed", "value": f"{last['elapsed_s']:.1f}s", "sub": f"{last['step']}/{last['total_steps']} ep", "tone": "neutral"},
        ],
        "charts": {
            "primary": {
                "title": "Validation BCE Loss",
                "unit": "",
                "x_label": "ep",
                "series": [_make_series("val_bce", "Val BCE", [{"x": r["step"], "y": r["val_bce"], "best": r["step"] == best_r["step"]} for r in rows], bold=True)],
            },
            "secondary": [
                {
                    "id": "accs",
                    "label": "Metrics %",
                    "title": "Recall, Precision, Aux & Value Sign (%)",
                    "unit": "%",
                    "y_min": 0,
                    "y_max": 100,
                    "series": [
                        _make_series("rec", "Recall", [{"x": r["step"], "y": r["rec"]} for r in rows], bold=True),
                        _make_series("prec", "Precision", [{"x": r["step"], "y": r["prec"]} for r in rows]),
                        _make_series("aux", "Aux Own", [{"x": r["step"], "y": r["aux"]} for r in rows]),
                        _make_series("vsign", "Value Sign", [{"x": r["step"], "y": r["vsign"]} for r in rows]),
                    ],
                }
            ],
        },
        "table": {
            "columns": ["Ep", "Val BCE", "Recall", "Prec", "Aux", "VSign"],
            "rows": [
                [f"{r['step']}/{r['total_steps']}", f"{r['val_bce']:.3f}", f"{r['rec']:.1f}%", f"{r['prec']:.1f}%", f"{r['aux']:.1f}%", f"{r['vsign']:.1f}%"]
                for r in reversed(rows[-15:])
            ],
        },
        "headline": f"val_bce {best_r['val_bce']:.3f} · aux {last['aux']:.1f}%",
    }


def _parse_jsonl(text: str, path: Path, label: str, mtime: float) -> Optional[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue  # gracefully skip partial/mid-write trailing JSON line
        if isinstance(obj, dict):
            records.append(obj)
    if not records:
        return None

    step_keys = ("step", "iter", "iteration", "epoch", "ep")
    step_label = "step"
    for sk in step_keys:
        if any(sk in r for r in records):
            step_label = "ep" if sk in ("epoch", "ep") else ("iter" if sk.startswith("iter") else "step")
            break

    total_steps = None
    title = None
    device = None
    elapsed_points: List[Tuple[int, float]] = []
    numeric_keys: List[str] = []
    meta_ignore = set(step_keys) | {
        "total_steps",
        "total_iters",
        "total_epochs",
        "elapsed_s",
        "time_s",
        "roll_s",
        "upd_s",
        "timestamp",
        "ts",
        "best",
        "is_best",
    }

    parsed_rows: List[Tuple[int, Dict[str, Any]]] = []
    for idx, rec in enumerate(records, start=1):
        step = idx
        for sk in step_keys:
            if isinstance(rec.get(sk), (int, float)):
                step = int(rec[sk])
                break
        for tk in ("total_steps", "total_iters", "total_epochs"):
            if isinstance(rec.get(tk), (int, float)):
                total_steps = int(rec[tk])
        title = title or rec.get("run_name") or rec.get("title")
        device = device or rec.get("device")
        el = rec.get("elapsed_s", rec.get("time_s"))
        if isinstance(el, (int, float)):
            elapsed_points.append((step, float(el)))
        for k, v in rec.items():
            if k not in meta_ignore and isinstance(v, (int, float)) and not isinstance(v, bool):
                if k not in numeric_keys:
                    numeric_keys.append(k)
        parsed_rows.append((step, rec))

    if not numeric_keys:
        return None

    primary_kws = ("eval", "val", "acc", "win", "wr", "score", "reward", "f1", "auc")
    primary_keys = [k for k in numeric_keys if any(w in k.lower() for w in primary_kws) and "loss" not in k.lower()]
    if not primary_keys:
        primary_keys = numeric_keys[:2]
    secondary_keys = [k for k in numeric_keys if k not in primary_keys]

    primary_series = []
    for idx, k in enumerate(primary_keys[:6]):
        pts = [
            {"x": st, "y": float(rec[k]), "best": bool(rec.get("best") or rec.get("is_best")) if idx == 0 else False}
            for st, rec in parsed_rows
            if isinstance(rec.get(k), (int, float)) and not isinstance(rec.get(k), bool)
        ]
        if pts:
            primary_series.append(_make_series(k, k, pts, bold=(idx == 0)))

    secondary = []
    for k in secondary_keys[:6]:
        pts = [
            {"x": st, "y": float(rec[k])}
            for st, rec in parsed_rows
            if isinstance(rec.get(k), (int, float)) and not isinstance(rec.get(k), bool)
        ]
        if pts:
            secondary.append(
                {
                    "id": k,
                    "label": k,
                    "title": k,
                    "unit": "",
                    "series": [_make_series(k, k, pts, bold=True)],
                }
            )

    last_step, last_rec = parsed_rows[-1]
    kpis = []
    for k in numeric_keys[:4]:
        vals = [(st, float(r[k])) for st, r in parsed_rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            cur_v = vals[-1][1]
            kpis.append({"label": k, "value": f"{cur_v:.4g}", "sub": f"{step_label} {vals[-1][0]}", "tone": "neutral"})

    has_timing = any(isinstance(r.get("roll_s"), (int, float)) for _, r in parsed_rows)
    timing_rows = [{"roll_s": r.get("roll_s"), "upd_s": r.get("upd_s")} for _, r in parsed_rows if isinstance(r.get("roll_s"), (int, float))]

    table_keys = numeric_keys[:5]
    return {
        "kind": "jsonl",
        "kind_label": "Metrics JSONL",
        "title": str(title) if title else label,
        "step": last_step,
        "total_steps": total_steps,
        "step_label": step_label,
        "elapsed_s": elapsed_points[-1][1] if elapsed_points else None,
        "elapsed_points": elapsed_points,
        "is_done": bool(last_rec.get("done")) or (total_steps is not None and last_step >= total_steps),
        "phase_note": None,
        "meta": {k: str(v) for k, v in [("device", device)] if v},
        "timing": _avg_timing(timing_rows) if has_timing else None,
        "kpis": kpis,
        "charts": {
            "primary": {
                "title": " / ".join(s["label"] for s in primary_series[:3]) or "Primary Metrics",
                "unit": "",
                "x_label": step_label,
                "series": primary_series,
            },
            "secondary": secondary,
        },
        "table": {
            "columns": [step_label.title(), *table_keys],
            "rows": [
                [str(st)] + [f"{rec[k]:.4g}" if isinstance(rec.get(k), (int, float)) else "—" for k in table_keys]
                for st, rec in reversed(parsed_rows[-15:])
            ],
        },
        "headline": " · ".join(f"{k} {last_rec[k]:.4g}" for k in numeric_keys[:2] if isinstance(last_rec.get(k), (int, float))),
    }


def _parse_generic_log(text: str, path: Path, label: str, mtime: float) -> Optional[Dict[str, Any]]:
    rows: List[Tuple[int, Optional[int], Dict[str, float]]] = []
    key_order: List[str] = []
    for ln in text.splitlines():
        sm = GEN_STEP_RE.search(ln)
        if not sm:
            continue
        kvs = GEN_KV_RE.findall(ln)
        if len(kvs) < 2:
            continue
        metrics: Dict[str, float] = {}
        for k, v, _pct in kvs:
            kl = k.lower()
            if kl in ("step", "iter", "iteration", "epoch", "ep", "lr"):
                if kl == "lr":
                    continue
            try:
                metrics[k] = float(v)
            except ValueError:
                continue
            if k not in key_order:
                key_order.append(k)
        if len(metrics) < 2:
            continue
        rows.append((int(sm.group(1)), int(sm.group(2)) if sm.group(2) else None, metrics))

    if len(rows) < 1 or not key_order:
        return None

    last_step, total_steps, last_m = rows[-1]
    primary_keys = key_order[:2]
    secondary_keys = key_order[2:6]
    return {
        "kind": "generic_log",
        "kind_label": "Generic Training Log",
        "title": label,
        "step": last_step,
        "total_steps": total_steps,
        "step_label": "step",
        "elapsed_s": None,
        "elapsed_points": [],
        "is_done": bool(total_steps and last_step >= total_steps),
        "phase_note": None,
        "meta": {},
        "timing": None,
        "kpis": [{"label": k, "value": f"{last_m[k]:.4g}", "sub": f"step {last_step}", "tone": "neutral"} for k in key_order[:4] if k in last_m],
        "charts": {
            "primary": {
                "title": " / ".join(primary_keys),
                "unit": "",
                "x_label": "step",
                "series": [
                    _make_series(k, k, [{"x": st, "y": m[k]} for st, _tot, m in rows if k in m], bold=(idx == 0))
                    for idx, k in enumerate(primary_keys)
                ],
            },
            "secondary": [
                {
                    "id": k,
                    "label": k,
                    "title": k,
                    "unit": "",
                    "series": [_make_series(k, k, [{"x": st, "y": m[k]} for st, _tot, m in rows if k in m], bold=True)],
                }
                for k in secondary_keys
            ],
        },
        "table": {
            "columns": ["Step", *key_order[:5]],
            "rows": [[str(st)] + [f"{m[k]:.4g}" if k in m else "—" for k in key_order[:5]] for st, _tot, m in reversed(rows[-15:])],
        },
        "headline": " · ".join(f"{k} {last_m[k]:.4g}" for k in key_order[:2] if k in last_m),
    }


def _avg_timing(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    rolls = [r["roll_s"] for r in rows if r.get("roll_s") is not None]
    upds = [r["upd_s"] for r in rows if r.get("upd_s") is not None]
    if not rolls and not upds:
        return None
    out: Dict[str, float] = {}
    if rolls:
        out["last_roll_s"] = rolls[-1]
        out["avg_roll_s"] = statistics.mean(rolls[-5:])
    if upds:
        out["last_upd_s"] = upds[-1]
        out["avg_upd_s"] = statistics.mean(upds[-5:])
    return out


PARSERS: List[Callable[[str, Path, str, float], Optional[Dict[str, Any]]]] = [
    _parse_orbit_ppo,
    _parse_orbit_softkd,
    _parse_galcon_ppo,
    _parse_galcon_sup,
    _parse_generic_log,
]


def parse_file(path: Path, label: str, now_s: float) -> Optional[Dict[str, Any]]:
    try:
        st = path.stat()
        text = _read_text_bounded(path)
    except OSError:
        return None

    parsed: Optional[Dict[str, Any]] = None
    if path.suffix == ".jsonl":
        parsed = _parse_jsonl(text, path, label, st.st_mtime)
    else:
        for fn in PARSERS:
            parsed = fn(text, path, label, st.st_mtime)
            if parsed is not None:
                break
    if parsed is None:
        return None

    age_s = max(0.0, now_s - st.st_mtime)
    status = "done" if parsed["is_done"] else ("live" if age_s <= LIVE_THRESHOLD_S else "stale")
    sec_per_step, eta_s = _compute_eta_s(
        status,
        parsed["step"],
        parsed["total_steps"],
        parsed.pop("elapsed_points", []),
        age_s,
    )
    tail = [ln[:300] for ln in text.splitlines() if ln.strip()][-25:]
    run_id = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]

    parsed.update(
        {
            "id": run_id,
            "status": status,
            "mtime_ms": int(st.st_mtime * 1000),
            "age_s": round(age_s, 1),
            "source": label,
            "path": str(path),
            "sec_per_step": round(sec_per_step, 1) if sec_per_step is not None else None,
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
            "tail": tail,
        }
    )
    parsed.pop("is_done", None)
    return parsed


def collect_runs(
    app_data: Path,
    workspaces: List[str],
    selected_id: Optional[str] = None,
    max_age_days: int = 14,
) -> Dict[str, Any]:
    """Scan logs and build the /api/runs JSON response payload."""
    now_s = time.time()
    ws_paths = _clean_workspace_paths(workspaces)
    sources = _discover_sources(app_data, ws_paths, max_age_days)

    runs: List[Dict[str, Any]] = []
    for path, label in sources:
        item = parse_file(path, label, now_s)
        if item is not None:
            runs.append(item)

    runs.sort(key=lambda r: (0 if r["status"] == "live" else 1, -r["mtime_ms"]))
    active = next((r for r in runs if r["id"] == selected_id), runs[0] if runs else None)

    summary_keys = (
        "id",
        "title",
        "kind",
        "kind_label",
        "status",
        "step",
        "total_steps",
        "step_label",
        "mtime_ms",
        "age_s",
        "source",
        "path",
        "headline",
    )
    summaries = [{k: r[k] for k in summary_keys} for r in runs]
    return {
        "generated_at_ms": int(now_s * 1000),
        "runs": summaries,
        "active": active,
    }


def _resolve_app_data(cli_arg: str) -> Path:
    """Resolve Antigravity appData dir from ANTIGRAVITY_EXECUTABLE_DATA_DIR or fallback."""
    exec_dir = os.environ.get("ANTIGRAVITY_EXECUTABLE_DATA_DIR", "").strip()
    if exec_dir:
        cur = Path(exec_dir).expanduser()
        while cur != cur.parent:
            if cur.name == "sidecar_data":
                return cur.parent
            cur = cur.parent
    return Path(cli_arg).expanduser()


def _env_workspaces() -> List[str]:
    """Extract workspace URIs/paths from Antigravity environment variables if set."""
    out: List[str] = []
    for var in ("ANTIGRAVITY_WORKSPACE_URIS", "ANTIGRAVITY_WORKSPACE_PATH", "ANTIGRAVITY_WORKSPACES"):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    out.extend(str(x) for x in parsed if isinstance(x, str))
                    continue
            except json.JSONDecodeError:
                pass
        out.append(raw)
    return out


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_standalone(port: int, app_data: Path, default_workspaces: List[str], max_age_days: int) -> None:
    """Run local read-only HTTP server for Antigravity sidecar UI and standalone preview."""
    static_dir = Path(__file__).resolve().parent
    files = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    }
    cache_lock = threading.Lock()
    cache: Dict[Tuple[Optional[str], Tuple[str, ...]], Tuple[float, bytes]] = {}
    cache_ttl_s = 1.5

    def get_runs_bytes(run_id: Optional[str], ws: List[str]) -> bytes:
        key = (run_id, tuple(ws))
        now = time.monotonic()
        with cache_lock:
            hit = cache.get(key)
            if hit is not None and (now - hit[0]) < cache_ttl_s:
                return hit[1]
        payload = json.dumps(collect_runs(app_data, ws, run_id, max_age_days)).encode("utf-8")
        with cache_lock:
            if len(cache) > 16:
                cache.clear()
            cache[key] = (now, payload)
        return payload

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, ctype: str, body: bytes, *, send_body: bool = True, no_store: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if send_body:
                self.wfile.write(body)

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path in ("/health", "/healthz"):
                self._send(200, "text/plain; charset=utf-8", b"ok\n", send_body=False)
                return
            if path == "/preload.js":
                body = b"// Pure-Python sidecar theme shim\n"
                self._send(200, "application/javascript; charset=utf-8", body, send_body=False)
                return
            if path in files:
                fname, ctype = files[path]
                body = (static_dir / fname).read_bytes()
                self._send(200, ctype, body, send_body=False)
                return
            self.send_error(404)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/health", "/healthz"):
                self._send(200, "text/plain; charset=utf-8", b"ok\n")
                return
            if parsed.path == "/preload.js":
                self._send(200, "application/javascript; charset=utf-8", b"// Pure-Python sidecar theme shim\n")
                return
            if parsed.path == "/api/runs":
                qs = parse_qs(parsed.query)
                run_id = qs.get("run", [None])[0]
                ws = qs.get("workspaces", []) + default_workspaces + _env_workspaces()
                payload = get_runs_bytes(run_id, ws)
                self._send(200, "application/json; charset=utf-8", payload, no_store=True)
                return
            if parsed.path in files:
                fname, ctype = files[parsed.path]
                body = (static_dir / fname).read_bytes()
                self._send(200, ctype, body)
                return
            self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    srv = _ReusableThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Training Monitor serving at http://127.0.0.1:{port}/", flush=True)
    srv.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description="Training Monitor log parser & server")
    ap.add_argument("--app-data", default=str(Path.home() / ".gemini" / "antigravity"))
    ap.add_argument("--workspaces", action="append", default=[])
    ap.add_argument("--run", default=None)
    ap.add_argument("--max-age-days", type=int, default=14)
    ap.add_argument("--serve", type=int, default=0)
    args = ap.parse_args()

    app_data = _resolve_app_data(args.app_data)
    if args.serve > 0:
        port = int(os.environ.get("ANTIGRAVITY_SIDECAR_WEB_PORT") or os.environ.get("PORT") or args.serve)
        serve_standalone(port, app_data, args.workspaces, args.max_age_days)
        return 0

    data = collect_runs(app_data, args.workspaces + _env_workspaces(), args.run, args.max_age_days)
    sys.stdout.write(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
