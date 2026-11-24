#!/usr/bin/env python3
"""Portable unit and HTTP tests for Training Monitor plugin."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

PLUGIN_DIR = Path(__file__).resolve().parents[1]
SIDECAR_DIR = PLUGIN_DIR / "sidecars" / "monitor"
sys.path.insert(0, str(SIDECAR_DIR))

import parser  # noqa: E402


def test_manifests() -> None:
    p_json = json.loads((PLUGIN_DIR / "plugin.json").read_text(encoding="utf-8"))
    allowed_keys = {
        "name", "description", "logo", "suggestedPrompts", "displayName",
        "version", "author", "homepage", "repository", "disabled",
    }
    assert set(p_json.keys()) <= allowed_keys, f"Unexpected keys: {set(p_json.keys()) - allowed_keys}"
    assert p_json["name"] == "training-monitor"
    assert (PLUGIN_DIR / p_json["logo"]).is_file()

    s_json = json.loads((SIDECAR_DIR / "sidecar.json").read_text(encoding="utf-8"))
    assert s_json["command"] == "python3"
    assert s_json["args"] == ["parser.py", "--serve", "8765"]
    assert s_json["has_web_ui"] is True
    assert s_json["ui_config"]["views"][0]["entrypoint"] == "SIDECAR_UI_ENTRYPOINT_AUX_PANE"
    assert s_json["ui_config"]["views"][0]["path"] == "/"


def test_all_log_formats() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tm_unit_"))
    try:
        app_data = tmp / "app"
        task_dir = app_data / "brain" / "11111111-1111-1111-1111-111111111111" / ".system_generated" / "tasks"
        task_dir.mkdir(parents=True)

        # 1. Orbit Wars PPO (live)
        (task_dir / "task-267.log").write_text(
            "[RL-League-PPO] Loaded checkpoints/planet_transformer_gen10_4m_base.pt | dev=dml | pool=['init'] | iters=30 pairs=24 (2P=48ep) workers=10 | lr=2.5e-05 clip=0.15\n"
            "  [Baseline EVAL(36g)] init=50.0% gen9_balanced=25.0% gen9_softkd=58.3% avg=44.4%\n"
            "  iter  1/30 ( 93.3s [roll=26.8s upd=19.7s]) | states=5717 | 2P= 31.2% (sw=3, sp=9) | rew=-0.38 kl=0.0032 akl=0.0032 ent=0.873 vl=0.037 aux=95.6%\n"
            "  iter  3/30 (223.0s [roll=28.0s upd=17.9s]) | states=6521 | 2P= 25.0% (sw=5, sp=2) | rew=-0.50 kl=0.0091 akl=0.0091 ent=0.887 vl=0.032 aux=95.3% | EVAL(36g): init=58.3% gen9_balanced=41.7% gen9_softkd=50.0% avg=50.0% *BEST*\n"
            "  iter  6/30 (403.1s [roll=29.3s upd=16.9s]) | states=5874 | 2P= 16.7% (sw=3, sp=2) | rew=-0.67 kl=0.0179 akl=0.0179 ent=1.075 vl=0.019 aux=95.1% | EVAL(36g): init=33.3% gen9_balanced=41.7% gen9_softkd=58.3% avg=44.4%\n",
            encoding="utf-8",
        )

        # 2. Orbit Wars Soft-KD (done)
        (task_dir / "task-255.log").write_text(
            "[Ensemble4M] Device=dml | teachers=['a.pt'] weights=[1.0]\n"
            "[SoftKD Train] Fresh init (d=256, h=8, L=7, ff=640) -> 4.35M params (device=dml) | train=21717 val=2961 | epochs=14 batch=256 lr=0.0008 | mid_w=2.0\n"
            "  ep  1/14 ( 27.1s, 27.1s/ep) | tr_kl=0.525 val_kl=0.340 | val_all= 77.7% val_launch= 32.5% val_tgt= 66.1% aux_own= 95.1% vsign= 72.0%\n"
            "  ep 14/14 (358.6s, 25.5s/ep) | tr_kl=0.043 val_kl=0.052 | val_all= 89.0% val_launch= 74.3% val_tgt= 88.2% aux_own= 96.4% vsign= 83.8%\n"
            "[SoftKD Train] Saved best 2P Specialist -> checkpoints/planet_transformer_gen10_4m_base.pt (17037.9 KB)\n"
            "[Ensemble4M Eval(32g)] gen9_balanced=31.2% gen9_softkd=25.0% avg=28.1%\n",
            encoding="utf-8",
        )

        # 3. Galcon Gen4 PPO
        (task_dir / "task-galcon.log").write_text(
            "[Galcon Gen4 Multi-Send Mirror PPO+LuaBoss(foo) x8w dml] Start 36g std+large val score = 58.3% (pool=2)\n"
            "[Galcon Gen4 Multi-Send PPO] iter  1/5 (eps=16,  8.1s [roll=3.9s upd=2.9s]) | wins=6/16 (sp=2/4, bots=2/4, boss=2/8, splits=3) | states=1100 | pool=2\n"
            "[Galcon Gen4 Multi-Send PPO] iter  2/5 (eps=32,  15.3s [roll=3.7s upd=2.8s]) | wins=5/16 (sp=2/4, bots=2/4, boss=1/8, splits=3) | states=1092 | pool=2 | val_36g=61.1% *BEST*\n",
            encoding="utf-8",
        )

        # 4. Galcon Supervised
        (task_dir / "task-gsup.log").write_text(
            "  [Galcon Gen4 Multi-Send] ep  1/4 (12.3s) | val_bce=0.412 rec= 71.0% prec= 82.4% aux= 91.2% vsign= 74.0%\n"
            "  [Galcon Gen4 Multi-Send] ep  2/4 (24.1s) | val_bce=0.351 rec= 75.5% prec= 85.1% aux= 93.0% vsign= 78.2%\n",
            encoding="utf-8",
        )

        # 5. Generic log
        (task_dir / "task-generic.log").write_text(
            "Epoch 1/10 | loss=0.912 val_loss=0.844 val_acc=68.5%\n"
            "Epoch 2/10 | loss=0.610 val_loss=0.580 val_acc=79.1%\n",
            encoding="utf-8",
        )

        # 6. Workspace metrics.jsonl (with trailing partial write)
        ws = tmp / "my_project"
        ws.mkdir()
        (ws / "metrics.jsonl").write_text(
            '{"step": 1, "total_steps": 10, "run_name": "CustomExp", "val_acc": 72.5, "loss": 0.62, "elapsed_s": 5.0}\n'
            '{"step": 2, "total_steps": 10, "run_name": "CustomExp", "val_acc": 78.9, "loss": 0.44, "elapsed_s": 10.2, "best": true}\n'
            '{"step": 3, "partial_unfinished_line":',
            encoding="utf-8",
        )

        # Unrelated log ignored
        (task_dir / "task-noise.log").write_text("pytest 31 passed in 1.2s\n", encoding="utf-8")

        res = parser.collect_runs(app_data, [f"file://{ws}"])
        kinds = {r["kind"]: r for r in res["runs"]}
        assert len(res["runs"]) == 6, f"Expected 6 runs, got {len(res['runs'])}"

        assert kinds["orbit_ppo"]["status"] == "live"
        assert kinds["orbit_ppo"]["step"] == 6 and kinds["orbit_ppo"]["total_steps"] == 30
        assert "best 50.0%" in kinds["orbit_ppo"]["headline"]
        assert kinds["orbit_softkd"]["status"] == "done" and kinds["orbit_softkd"]["step"] == 14
        assert kinds["galcon_ppo"]["status"] == "live" and "val 61.1%" in kinds["galcon_ppo"]["headline"]
        assert kinds["galcon_sup"]["status"] == "live"
        assert kinds["jsonl"]["title"] == "CustomExp" and kinds["jsonl"]["step"] == 2
        assert kinds["generic_log"]["step"] == 2

        detail = parser.collect_runs(app_data, [str(ws)], selected_id=kinds["orbit_ppo"]["id"])["active"]
        assert detail["eta_s"] is not None and detail["eta_s"] > 0
        assert detail["timing"]["last_roll_s"] == 29.3
        avg_series = next(s for s in detail["charts"]["primary"]["series"] if s["key"] == "avg")
        assert [p["x"] for p in avg_series["points"]] == [0, 3, 6]
        assert avg_series["points"][1]["best"] is True

        for f in (SIDECAR_DIR / "index.html", SIDECAR_DIR / "styles.css", SIDECAR_DIR / "app.js"):
            assert f.is_file() and f.stat().st_size > 100

        env = {**os.environ, "ANTIGRAVITY_SIDECAR_WEB_PORT": "18767"}
        proc = subprocess.Popen(
            [sys.executable, str(SIDECAR_DIR / "parser.py"), "--app-data", str(app_data), "--workspaces", str(ws), "--serve", "8765"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            for _ in range(25):
                try:
                    health = urllib.request.urlopen("http://127.0.0.1:18767/health", timeout=1)
                    assert health.status == 200 and health.read().decode("utf-8").strip() == "ok"

                    head_req = urllib.request.Request("http://127.0.0.1:18767/", method="HEAD")
                    assert urllib.request.urlopen(head_req, timeout=1).status == 200

                    html = urllib.request.urlopen("http://127.0.0.1:18767/?token=test", timeout=1).read().decode("utf-8")
                    assert "Training Monitor" in html and 'id="btn-theme"' in html
                    css = urllib.request.urlopen("http://127.0.0.1:18767/styles.css", timeout=1).read().decode("utf-8")
                    assert "color-scheme: dark;" in css and "#111315" in css
                    api_res = json.loads(urllib.request.urlopen("http://127.0.0.1:18767/api/runs", timeout=1).read().decode("utf-8"))
                    assert len(api_res["runs"]) == 6
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise AssertionError("Standalone HTTP server failed to respond on ANTIGRAVITY_SIDECAR_WEB_PORT")
        finally:
            proc.terminate()
            proc.wait(timeout=3)

        print("ALL UNIT TESTS PASSED:", sorted(kinds.keys()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_manifests()
    test_all_log_formats()
