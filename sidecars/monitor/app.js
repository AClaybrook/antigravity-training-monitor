/* Training Monitor side-pane UI. Zero external dependencies, theme-synced via /preload.js. */
(() => {
  const POLL_MS = 3000;
  const COLORS = ['var(--c-0)', 'var(--c-1)', 'var(--c-2)', 'var(--c-3)', 'var(--c-4)', 'var(--c-5)'];
  const $ = (id) => document.getElementById(id);

  const state = {
    paused: false,
    selectedRun: '',
    secondaryId: '',
    hiddenSeries: new Set(),
    workspaces: [],
    lastData: null,
    timer: null,
  };

  const SUN_SVG = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="3"/><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M12.6 3.4l-1.1 1.1M4.5 11.5l-1.1 1.1"/></svg>';
  const MOON_SVG = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M13.2 9.8A5.4 5.4 0 0 1 6.2 2.8a5.5 5.5 0 1 0 7 7z"/></svg>';

  function getSavedTheme() {
    try {
      const t = localStorage.getItem('tm-theme');
      return t === 'dark' || t === 'light' ? t : null;
    } catch {
      return null;
    }
  }

  function applyTheme(theme, manual = false) {
    if (manual) document.documentElement.dataset.themeManual = '1';
    else delete document.documentElement.dataset.themeManual;
    if (document.documentElement.dataset.theme !== theme) document.documentElement.dataset.theme = theme;
    const btn = $('btn-theme');
    if (btn) {
      btn.innerHTML = theme === 'dark' ? SUN_SVG : MOON_SVG;
      btn.title = theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
    }
  }

  function syncScheme() {
    const saved = getSavedTheme();
    if (saved) {
      applyTheme(saved, true);
      return;
    }
    const hostBg = getComputedStyle(document.documentElement).getPropertyValue('--background').trim();
    if (!hostBg) {
      applyTheme('dark', false);
      return;
    }
    const probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;color:var(--background)';
    document.documentElement.appendChild(probe);
    const c = getComputedStyle(probe).color;
    probe.remove();
    const nums = (c.match(/[\d.]+/g) || []).map(Number);
    if (nums.length < 3) {
      applyTheme('dark', false);
      return;
    }
    const scale = c.startsWith('color(') ? 1 : 255;
    const [r, g, b] = nums.slice(0, 3).map((v) => v / scale);
    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    applyTheme(lum < 0.5 ? 'dark' : 'light', false);
  }
  syncScheme();
  new MutationObserver(syncScheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['style'],
  });
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', syncScheme);
  window.addEventListener('message', (ev) => {
    const d = ev?.data;
    if (!d || typeof d !== 'object') return;
    const tokens = d.tokens || d.styles || (typeof d['--background'] === 'string' ? d : null);
    if (tokens && typeof tokens === 'object') {
      for (const [k, v] of Object.entries(tokens)) {
        if (k.startsWith('--') && typeof v === 'string') document.documentElement.style.setProperty(k, v);
      }
      syncScheme();
    } else if (!getSavedTheme() && (d.theme === 'dark' || d.theme === 'light')) {
      applyTheme(d.theme, false);
    }
  });

  const esc = (s) =>
    String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function fmtDur(sec) {
    if (!Number.isFinite(sec) || sec < 0) return '—';
    const s = Math.round(sec);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
    return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
  }

  function fmtNum(v, unit = '') {
    if (!Number.isFinite(v)) return '—';
    if (unit === '%') return `${v.toFixed(1)}%`;
    if (unit === 's') return `${v.toFixed(1)}s`;
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    if (Math.abs(v) >= 0.1) return v.toFixed(3);
    return v.toFixed(4);
  }

  async function loadWorkspaces() {
    try {
      const fn = window.sidecar?.getWorkspaceUris;
      if (typeof fn !== 'function') return;
      const res = await Promise.resolve(fn.call(window.sidecar));
      if (Array.isArray(res)) state.workspaces = res.filter((x) => typeof x === 'string');
    } catch {
      // Ignore if running standalone without window.sidecar
    }
  }

  async function apiGet(path) {
    const f = window.sidecar?.fetch ? window.sidecar.fetch.bind(window.sidecar) : fetch;
    const res = await f(path, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function renderChart(containerEl, chartSpec, hiddenKeys, unitOverride) {
    const allSeries = (chartSpec?.series || []).filter((s) => Array.isArray(s.points) && s.points.length > 0);
    if (!allSeries.length) {
      containerEl.innerHTML = '<div class="chart-empty">Waiting for first data point…</div>';
      return;
    }

    const colorMap = new Map(allSeries.map((s, idx) => [s.key, COLORS[idx % COLORS.length]]));
    const visible = allSeries.filter((s) => !hiddenKeys.has(s.key));
    if (!visible.length) {
      containerEl.innerHTML = '<div class="chart-empty">All series hidden — click a legend item to show.</div>';
      return;
    }

    const W = 420;
    const H = 176;
    const pad = { t: 10, r: 10, b: 24, l: 40 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const xs = visible.flatMap((s) => s.points.map((p) => p.x));
    const ys = visible.flatMap((s) => s.points.map((p) => p.y));
    let minX = Math.min(...xs);
    let maxX = Math.max(...xs);
    if (minX === maxX) { minX -= 1; maxX += 1; }

    let minY = Number.isFinite(chartSpec.y_min) ? chartSpec.y_min : Math.min(...ys);
    let maxY = Number.isFinite(chartSpec.y_max) ? chartSpec.y_max : Math.max(...ys);
    if (Number.isFinite(chartSpec.ref_line)) {
      minY = Math.min(minY, chartSpec.ref_line);
      maxY = Math.max(maxY, chartSpec.ref_line);
    }
    if (minY === maxY) {
      const d = Math.max(Math.abs(minY) * 0.1, 1);
      minY -= d;
      maxY += d;
    } else if (!Number.isFinite(chartSpec.y_min) || !Number.isFinite(chartSpec.y_max)) {
      const span = maxY - minY;
      if (!Number.isFinite(chartSpec.y_min)) minY = Math.max(0, minY - span * 0.08);
      if (!Number.isFinite(chartSpec.y_max)) maxY += span * 0.08;
    }

    const sx = (x) => pad.l + ((x - minX) / (maxX - minX)) * pw;
    const sy = (y) => pad.t + ph - ((y - minY) / (maxY - minY)) * ph;
    const unit = unitOverride ?? chartSpec.unit ?? '';

    const parts = [`<svg class="chart-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`];

    // Gridlines + Y ticks
    const ticks = 4;
    for (let i = 0; i <= ticks; i++) {
      const frac = i / ticks;
      const val = minY + (maxY - minY) * frac;
      const py = sy(val).toFixed(1);
      parts.push(`<line x1="${pad.l}" y1="${py}" x2="${W - pad.r}" y2="${py}" stroke="var(--tm-border)" stroke-width="1"/>`);
      parts.push(`<text x="${pad.l - 5}" y="${Number(py) + 3}" text-anchor="end" fill="var(--tm-fg-muted)" font-size="9.5" font-family="var(--tm-mono)">${esc(fmtNum(val, unit))}</text>`);
    }

    // X ticks
    const uniqueX = [...new Set(xs)].sort((a, b) => a - b);
    const xTicks = uniqueX.length <= 5 ? uniqueX : [uniqueX[0], uniqueX[Math.floor(uniqueX.length / 2)], uniqueX[uniqueX.length - 1]];
    for (const xv of xTicks) {
      const px = sx(xv).toFixed(1);
      parts.push(`<text x="${px}" y="${H - 6}" text-anchor="middle" fill="var(--tm-fg-muted)" font-size="9.5" font-family="var(--tm-mono)">${xv}</text>`);
    }

    // Optional reference line (e.g. 50%)
    if (Number.isFinite(chartSpec.ref_line) && chartSpec.ref_line >= minY && chartSpec.ref_line <= maxY) {
      const ry = sy(chartSpec.ref_line).toFixed(1);
      parts.push(`<line x1="${pad.l}" y1="${ry}" x2="${W - pad.r}" y2="${ry}" stroke="var(--tm-fg-muted)" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="4 4"/>`);
    }

    // Polylines + points
    for (const s of visible) {
      const col = colorMap.get(s.key) || COLORS[0];
      const d = s.points.map((p, idx) => `${idx === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
      parts.push(`<path d="${d}" fill="none" stroke="${col}" stroke-width="${s.bold ? 2.2 : 1.5}" stroke-linejoin="round" stroke-linecap="round" stroke-opacity="${s.bold ? 1 : 0.8}"/>`);
      for (const p of s.points) {
        const cx = sx(p.x).toFixed(1);
        const cy = sy(p.y).toFixed(1);
        if (p.best) {
          parts.push(`<circle cx="${cx}" cy="${cy}" r="5" fill="none" stroke="var(--tm-warning)" stroke-width="1.6"/>`);
          parts.push(`<circle cx="${cx}" cy="${cy}" r="3" fill="${col}"/>`);
        } else if (s.points.length <= 24) {
          parts.push(`<circle cx="${cx}" cy="${cy}" r="${s.bold ? 2.5 : 1.9}" fill="${col}"/>`);
        }
      }
    }

    // Hover indicator line + transparent hit overlay
    parts.push(`<line class="hover-line hidden" x1="0" y1="${pad.t}" x2="0" y2="${H - pad.b}" stroke="var(--tm-fg-muted)" stroke-opacity="0.55" stroke-width="1" stroke-dasharray="2 2"/>`);
    parts.push(`<rect class="hit-rect" x="${pad.l}" y="${pad.t}" width="${pw}" height="${ph}" fill="transparent"/>`);
    parts.push('</svg>');
    containerEl.innerHTML = parts.join('');

    const svg = containerEl.querySelector('svg');
    const hit = svg.querySelector('.hit-rect');
    const hoverLine = svg.querySelector('.hover-line');
    const tooltip = $('tooltip');

    const hideTip = () => {
      hoverLine.classList.add('hidden');
      tooltip.classList.add('hidden');
    };

    hit.addEventListener('pointermove', (ev) => {
      const rect = svg.getBoundingClientRect();
      const svgX = ((ev.clientX - rect.left) / Math.max(1, rect.width)) * W;
      const targetX = minX + ((svgX - pad.l) / pw) * (maxX - minX);
      let nearestX = uniqueX[0];
      let bestDist = Infinity;
      for (const xv of uniqueX) {
        const dist = Math.abs(xv - targetX);
        if (dist < bestDist) {
          bestDist = dist;
          nearestX = xv;
        }
      }
      const lineX = sx(nearestX).toFixed(1);
      hoverLine.setAttribute('x1', lineX);
      hoverLine.setAttribute('x2', lineX);
      hoverLine.classList.remove('hidden');

      const rows = [];
      let hasBest = false;
      for (const s of visible) {
        const pt = s.points.find((p) => p.x === nearestX);
        if (!pt) continue;
        if (pt.best) hasBest = true;
        rows.push(
          `<div class="tooltip-row"><span><span class="legend-swatch" style="display:inline-block;background:${colorMap.get(s.key)}"></span> ${esc(s.label)}</span><strong>${esc(fmtNum(pt.y, unit))}${pt.best ? ' ★' : ''}</strong></div>`
        );
      }
      if (!rows.length) return hideTip();
      const xTitle = `${chartSpec.x_label || 'step'} ${nearestX}${hasBest ? ' · BEST' : ''}`;
      tooltip.innerHTML = `<div class="tooltip-title">${esc(xTitle)}</div>${rows.join('')}`;
      tooltip.classList.remove('hidden');

      const tx = Math.min(window.innerWidth - tooltip.offsetWidth - 10, Math.max(8, ev.clientX + 12));
      const ty = Math.min(window.innerHeight - tooltip.offsetHeight - 10, Math.max(8, ev.clientY - 12));
      tooltip.style.left = `${tx}px`;
      tooltip.style.top = `${ty}px`;
    });
    hit.addEventListener('pointerleave', hideTip);
  }

  function renderLegend(legendEl, chartSpec, scopeKey) {
    const series = (chartSpec?.series || []).filter((s) => s.points?.length);
    legendEl.innerHTML = series
      .map((s, idx) => {
        const col = COLORS[idx % COLORS.length];
        const keyId = `${scopeKey}:${s.key}`;
        const muted = state.hiddenSeries.has(keyId) ? ' muted' : '';
        return `<button type="button" class="legend-item${muted}" data-series-key="${esc(keyId)}"><span class="legend-swatch" style="background:${col}"></span>${esc(s.label)}</button>`;
      })
      .join('');
    legendEl.querySelectorAll('.legend-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        const k = btn.dataset.seriesKey;
        if (state.hiddenSeries.has(k)) state.hiddenSeries.delete(k);
        else state.hiddenSeries.add(k);
        if (state.lastData?.active) renderActive(state.lastData.active);
      });
    });
  }

  function renderActive(run) {
    if (!run) {
      $('dashboard').classList.add('hidden');
      $('empty-state').classList.remove('hidden');
      $('status-badge').classList.add('hidden');
      return;
    }
    $('empty-state').classList.add('hidden');
    $('dashboard').classList.remove('hidden');

    const badge = $('status-badge');
    badge.className = `badge badge-${run.status === 'live' ? 'live' : run.status === 'done' ? 'done' : 'stale'}`;
    $('status-text').textContent = run.status === 'live' ? 'Live' : run.status === 'done' ? 'Done' : 'Stale';

    const metaBits = [run.source, `updated ${fmtDur(run.age_s)} ago`];
    for (const [mk, mv] of Object.entries(run.meta || {})) metaBits.push(`${mk}=${mv}`);
    $('run-sub').innerHTML = metaBits.map((b) => `<span>${esc(b)}</span>`).join('');

    const pct = run.total_steps ? Math.min(100, Math.max(0, (100 * run.step) / run.total_steps)) : 0;
    const stepText = run.total_steps ? `${run.step_label} ${run.step} / ${run.total_steps} (${pct.toFixed(0)}%)` : `${run.step_label} ${run.step}`;
    $('step-label').textContent = stepText;

    const pnote = $('phase-note');
    if (run.phase_note) {
      pnote.textContent = run.phase_note;
      pnote.classList.remove('hidden');
    } else {
      pnote.classList.add('hidden');
    }

    if (run.status === 'done') {
      $('eta-label').textContent = run.elapsed_s ? `Finished in ${fmtDur(run.elapsed_s)}` : 'Completed';
    } else if (Number.isFinite(run.eta_s)) {
      $('eta-label').textContent = `ETA ~${fmtDur(run.eta_s)}`;
    } else if (Number.isFinite(run.elapsed_s)) {
      $('eta-label').textContent = `Elapsed ${fmtDur(run.elapsed_s)}`;
    } else {
      $('eta-label').textContent = run.status === 'live' ? 'Running…' : 'Stopped';
    }

    const fill = $('progress-fill');
    fill.style.width = `${run.total_steps ? pct.toFixed(1) : (run.status === 'done' ? 100 : 8)}%`;
    fill.classList.toggle('done', run.status === 'done');

    const tRow = $('timing-row');
    const timingParts = [];
    if (Number.isFinite(run.elapsed_s)) timingParts.push(`<span>Elapsed ${esc(fmtDur(run.elapsed_s))}</span>`);
    if (Number.isFinite(run.sec_per_step)) timingParts.push(`<span>~${run.sec_per_step.toFixed(1)}s / ${esc(run.step_label)}</span>`);
    if (run.timing?.avg_roll_s && run.timing?.avg_upd_s) {
      const tot = run.timing.avg_roll_s + run.timing.avg_upd_s;
      const rp = Math.round((100 * run.timing.avg_roll_s) / tot);
      tRow.innerHTML =
        `<div>${timingParts.join(' · ')}</div>` +
        `<div class="split-bar"><span>roll ${run.timing.avg_roll_s.toFixed(1)}s</span><span class="split-track"><span class="split-roll" style="width:${rp}%"></span><span class="split-upd" style="width:${100 - rp}%"></span></span><span>upd ${run.timing.avg_upd_s.toFixed(1)}s</span></div>`;
      tRow.classList.remove('hidden');
    } else if (timingParts.length) {
      tRow.innerHTML = `<div>${timingParts.join(' · ')}</div>`;
      tRow.classList.remove('hidden');
    } else {
      tRow.classList.add('hidden');
    }

    $('kpi-grid').innerHTML = (run.kpis || [])
      .map(
        (k) => `<div class="kpi-card"><div class="kpi-label">${esc(k.label)}</div><div class="kpi-value tone-${esc(k.tone || 'neutral')}">${esc(k.value)}</div><div class="kpi-sub">${esc(k.sub || '')}</div></div>`
      )
      .join('');

    const primary = run.charts?.primary;
    $('primary-title').textContent = primary?.title || 'Primary Metric';
    renderLegend($('primary-legend'), primary, 'p');
    const hiddenPrimary = new Set([...state.hiddenSeries].filter((k) => k.startsWith('p:')).map((k) => k.slice(2)));
    renderChart($('primary-chart'), primary, hiddenPrimary);

    const secs = run.charts?.secondary || [];
    const secCard = $('secondary-card');
    if (!secs.length) {
      secCard.classList.add('hidden');
    } else {
      secCard.classList.remove('hidden');
      const curSec = secs.find((s) => s.id === state.secondaryId) || secs[0];
      state.secondaryId = curSec.id;
      $('secondary-tabs').innerHTML = secs
        .map((s) => `<button type="button" class="subtab${s.id === curSec.id ? ' active' : ''}" data-sec-id="${esc(s.id)}">${esc(s.label)}</button>`)
        .join('');
      $('secondary-tabs').querySelectorAll('.subtab').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.secondaryId = btn.dataset.secId;
          renderActive(state.lastData.active);
        });
      });
      $('secondary-sub').textContent = curSec.title || '';
      const secScope = `s:${curSec.id}`;
      renderLegend($('secondary-legend'), curSec, secScope);
      const hiddenSec = new Set(
        [...state.hiddenSeries].filter((k) => k.startsWith(`${secScope}:`)).map((k) => k.slice(secScope.length + 1)),
      );
      renderChart($('secondary-chart'), { ...curSec, x_label: primary?.x_label || 'step' }, hiddenSec);
    }

    const tbl = run.table;
    if (tbl?.rows?.length) {
      $('table-drawer').classList.remove('hidden');
      $('table-count').textContent = `last ${tbl.rows.length}`;
      $('steps-table').innerHTML =
        `<thead><tr>${tbl.columns.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead>` +
        `<tbody>${tbl.rows.map((r) => `<tr>${r.map((cell) => `<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}</tbody>`;
    } else {
      $('table-drawer').classList.add('hidden');
    }

    $('tail-path').textContent = run.source || '';
    $('log-tail').textContent = (run.tail || []).join('\n');
  }

  function renderRunSelector(runs, active) {
    const wrap = $('selector-wrap');
    if (!runs.length) {
      wrap.classList.add('hidden');
      return;
    }
    wrap.classList.remove('hidden');
    const sel = $('run-select');
    const activeId = active?.id || runs[0].id;
    state.selectedRun = activeId;
    sel.innerHTML = runs
      .map((r) => {
        const tag = r.status === 'live' ? 'LIVE' : r.status === 'done' ? 'DONE' : 'IDLE';
        const prog = r.total_steps ? `${r.step}/${r.total_steps}` : `${r.step_label} ${r.step}`;
        const hl = r.headline ? ` · ${r.headline}` : '';
        return `<option value="${esc(r.id)}"${r.id === activeId ? ' selected' : ''}>[${tag}] ${esc(r.title)} (${esc(prog)}${esc(hl)})</option>`;
      })
      .join('');
  }

  async function refresh() {
    try {
      const qs = new URLSearchParams();
      if (state.selectedRun) qs.set('run', state.selectedRun);
      if (state.workspaces.length) qs.set('workspaces', state.workspaces.join(','));
      const data = await apiGet(`/api/runs?${qs.toString()}`);
      state.lastData = data;
      if (data.error) {
        $('error-banner').textContent = data.error;
        $('error-banner').classList.remove('hidden');
      } else {
        $('error-banner').classList.add('hidden');
      }
      renderRunSelector(data.runs || [], data.active);
      renderActive(data.active);
      $('poll-meta').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (err) {
      $('error-banner').textContent = `Monitor poll error: ${err?.message || err}`;
      $('error-banner').classList.remove('hidden');
    }
  }

  function schedulePoll() {
    if (state.timer) clearInterval(state.timer);
    state.timer = setInterval(() => {
      if (!state.paused && !document.hidden) refresh();
    }, POLL_MS);
  }

  $('run-select').addEventListener('change', (ev) => {
    state.selectedRun = ev.target.value;
    state.hiddenSeries.clear();
    refresh();
  });

  $('btn-refresh').addEventListener('click', () => refresh());

  $('btn-theme').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem('tm-theme', next);
    } catch {
      // Ignore storage failures in sandboxed iframes
    }
    applyTheme(next, true);
  });

  $('btn-pause').addEventListener('click', () => {
    state.paused = !state.paused;
    const btn = $('btn-pause');
    btn.classList.toggle('active', state.paused);
    btn.title = state.paused ? 'Resume auto-refresh' : 'Pause auto-refresh';
    if (!state.paused) refresh();
  });

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !state.paused) refresh();
  });

  (async () => {
    await loadWorkspaces();
    await refresh();
    schedulePoll();
  })();
})();
