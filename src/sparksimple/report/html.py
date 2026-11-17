"""sparksimple.report.html — HTML report generator for a Phase 10 analysis result."""

from __future__ import annotations

import math
import html as _html
from pathlib import Path
from typing import Any

import numpy as np

from sparksimple.export.pipeline import Phase10Result
from sparksimple.report.figures import (
    generate_blank,
    generate_fo_map,
    generate_location_sparks,
    generate_location_sparks2,
    generate_location_rois,
    generate_mask_image,
    generate_spark_feature_plot,
    generate_roi_signal,
    generate_detection_film,
    generate_fo_map_raw,
    generate_roi_heatmap,
)


_MISSING = object()


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    v = getattr(obj, attr, _MISSING)
    if v is not _MISSING:
        return v
    feature = getattr(obj, "feature", None)
    if feature is not None:
        v2 = getattr(feature, attr, _MISSING)
        if v2 is not _MISSING:
            return v2
    return default


def _fmt(v: Any, decimals: int = 2) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _esc(s: Any) -> str:
    return _html.escape(str(s))


# ── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:        #f0f7ff;
  --surface:   #ffffff;
  --border:    #bfdbfe;
  --text:      #1e293b;
  --muted:     #64748b;
  --accent:    #0369a1;
  --green-bg:  #f0fdf4;
  --green-hd:  #15803d;
  --blue-bg:   #eff6ff;
  --blue-hd:   #1d4ed8;
  --red-bg:    #fff1f2;
  --red-hd:    #be123c;
  --row-sel:   rgba(3,105,161,0.12);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { background: var(--bg); color: var(--text); font-size: 12px; }

/* ── Header ── */
#hdr {
  position: sticky; top: 0; z-index: 100;
  background: var(--surface);
  border-bottom: 2px solid var(--accent);
  padding: 6px 12px;
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
#hdr h1 { font-size: 14px; font-weight: 700; color: var(--accent); white-space: nowrap; }
.kpi-bar { display: flex; flex-wrap: wrap; gap: 10px; flex: 1; }
.kpi { display: flex; flex-direction: column; align-items: center; min-width: 64px; }
.kpi-lbl { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; }
.kpi-val { font-size: 12px; font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; }

/* ── ROI signal strip (full-width, edge-to-edge) ── */
#roi-strip {
  display: none;
  width: 100%;
  background: var(--surface);
  border-bottom: 2px solid var(--accent);
}
#roi-strip.active { display: block; }
#roi-strip-img {
  width: 100%;
  height: auto;
  display: block;
  cursor: zoom-out;
}

/* ── Main 3-column layout ── */
#main {
  display: flex;
  height: calc(100vh - 50px);
  overflow: hidden;
}
.col-left   { flex: 0 0 46%; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid var(--border); }
.col-centre { flex: 0 0 27%; display: flex; flex-direction: column; overflow-y: auto; border-right: 1px solid var(--border); background: var(--bg); padding: 6px; }
.col-right  { flex: 0 0 27%; display: flex; flex-direction: column; overflow-y: auto; background: var(--bg); padding: 6px; }

/* ── Table panels ── */
.tbl-panel { display: flex; flex-direction: column; flex: 1; overflow: hidden; min-height: 120px; }
.tbl-panel + .tbl-panel { border-top: 1px solid var(--border); }
.tbl-hd {
  font-size: 10px; font-weight: 700; padding: 4px 8px; color: #fff;
  display: flex; align-items: center; gap: 6px; cursor: default;
  user-select: none; flex-shrink: 0;
}
.tbl-hd .count { font-weight: 400; opacity: .75; }
.tbl-wrap { overflow: auto; flex: 1; }
table { width: 100%; border-collapse: collapse; }
th {
  position: sticky; top: 0; z-index: 1;
  font-size: 10px; padding: 3px 5px; white-space: nowrap; cursor: pointer;
  background: var(--surface); border-bottom: 1px solid var(--border);
  text-align: right; color: var(--muted);
  user-select: none;
}
th:first-child { text-align: left; }
th.sorted-asc::after  { content: " ↑"; color: var(--accent); }
th.sorted-desc::after { content: " ↓"; color: var(--accent); }
td { font-size: 11px; padding: 2px 5px; white-space: nowrap; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--border); text-align: right; }
td:first-child { text-align: left; }
tr.selected td { background: var(--row-sel); }
tr:hover td { opacity: .85; }

/* panel colours */
.panel-green .tbl-hd { background: var(--green-hd); }
.panel-green         { background: var(--green-bg); }
.panel-blue  .tbl-hd { background: var(--blue-hd); }
.panel-blue          { background: var(--blue-bg); }
.panel-red   .tbl-hd { background: var(--red-hd); }
.panel-red           { background: var(--red-bg); }

/* spark / ROI buttons */
.spk-btn, .roi-btn {
  padding: 1px 6px; border-radius: 3px; cursor: pointer; font-size: 10px;
  border: 1px solid currentColor; background: transparent;
  font-variant-numeric: tabular-nums;
}
.spk-btn { color: #15803d; }
.spk-btn:hover { background: rgba(21,128,61,.13); }
.roi-btn { color: #1d4ed8; }
.roi-btn:hover { background: rgba(29,78,216,.12); }

/* ── Centre panel ── */
.centre-title { font-size: 10px; color: var(--muted); padding: 2px 4px; text-align: center; }
.centre-pair { display: flex; flex-direction: column; gap: 4px; flex: 1; min-height: 0; }
.centre-pair img {
  max-width: 100%; max-height: 100%; width: auto; height: auto;
  object-fit: contain; margin: 0 auto; display: block;
  border-radius: 4px; border: 1px solid var(--border); background: #fff;
}

/* ── Right panel ── */
.right-section { margin-bottom: 8px; }
.right-section h3 { font-size: 10px; color: var(--muted); margin-bottom: 4px; }
.right-section img { width: 100%; border-radius: 4px; border: 1px solid var(--border); background: #fff; }
#film-ctrl { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
#film-ctrl button {
  padding: 2px 8px; cursor: pointer;
  background: var(--accent); color: #fff;
  border: none; border-radius: 3px; font-size: 11px;
}
#frame-counter { font-size: 10px; color: var(--muted); flex: 1; text-align: center; font-variant-numeric: tabular-nums; }
#frame-img { border-radius: 4px; border: 1px solid var(--border); background: var(--surface); min-height: 60px; }

/* ── Interactive ROI map ── */
#roi-map-wrap {
  position: relative; display: block; line-height: 0;
  border-radius: 4px; overflow: hidden; border: 1px solid var(--border);
}
#roi-map-bg { width: 100%; height: 100%; display: block; background: #000; object-fit: fill; }
#roi-map-svg {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
}
.roi-marker { cursor: pointer; }
.roi-marker circle {
  fill: var(--roi-color, #93c5fd);
  fill-opacity: 0.15;
  stroke: var(--roi-color, #93c5fd);
  stroke-width: 1.5px;
  transition: fill-opacity .15s, stroke-width .15s;
}
.roi-marker:hover circle { fill-opacity: 0.32; stroke-width: 2.2px; }
.roi-marker.roi-selected circle {
  fill: #fb923c; stroke: #fb923c; fill-opacity: 0.28; stroke-width: 2.5px;
}
.roi-marker text {
  fill: #ffffff; font-weight: 700; text-anchor: middle; dominant-baseline: central;
  pointer-events: none; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.roi-marker.roi-selected text { fill: #fff7ed; }
/* tooltip */
#roi-tooltip {
  display: none; position: absolute; z-index: 50; pointer-events: none;
  background: rgba(15,23,42,0.92); color: #f1f5f9;
  padding: 5px 8px; border-radius: 5px; font-size: 10px; line-height: 1.6;
  white-space: nowrap; box-shadow: 0 2px 8px rgba(0,0,0,.45);
}
/* color legend — vertical strip to the right of the map */
.roi-map-outer { display: grid; grid-template-columns: minmax(0, 1fr) 22px; align-items: start; gap: 4px; }
.roi-map-outer #roi-map-wrap { min-width: 0; }
.roi-legend-vert { width: 22px; height: 100%; display: flex; }
.roi-legend-vert svg { width: 100%; height: 100%; display: block; overflow: visible; }
"""

_JS = r"""
let cur = 1;
const totalFrames = FILM_TOTAL;
const sparkFrames = SPARK_FRAMES;
const sparkToROI = SPARK_TO_ROI;
const roiToFirstSpark = ROI_TO_FIRST_SPARK;
let selSpkRow = null;
let selRoiRow = null;
let selRoiMarker = null;

function highlightROIMarker(id) {
  if (selRoiMarker) selRoiMarker.classList.remove('roi-selected');
  selRoiMarker = document.getElementById('roi-marker-' + id);
  if (selRoiMarker) selRoiMarker.classList.add('roi-selected');
}

function selectSpark(id, syncROI = true, doScroll = true) {
  document.getElementById('ctr-feat').src = 'spkFeat/Spk' + id + '.png';
  document.getElementById('ctr-title').textContent = 'Spark ' + id;
  if (selSpkRow) selSpkRow.classList.remove('selected');
  selSpkRow = document.getElementById('spkrow-' + id);
  if (selSpkRow) selSpkRow.classList.add('selected');
  if (sparkFrames[id]) {
    setFrame(sparkFrames[id]);
    if (doScroll) document.getElementById('frame-img').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  if (syncROI && sparkToROI[id]) selectROI(sparkToROI[id], false);
}

function selectROI(id, syncSpark = true) {
  const strip = document.getElementById('roi-strip');
  strip.classList.add('active');
  document.getElementById('roi-strip-img').src = 'roiFeat/ROI' + id + '.jpg';
  if (selRoiRow) selRoiRow.classList.remove('selected');
  selRoiRow = document.getElementById('roirow-' + id);
  if (selRoiRow) selRoiRow.classList.add('selected');
  highlightROIMarker(id);
  if (syncSpark && roiToFirstSpark[id]) selectSpark(roiToFirstSpark[id], false, false);
}

document.getElementById('roi-strip-img').addEventListener('dblclick', function() {
  document.getElementById('roi-strip').classList.remove('active');
  this.src = 'extraFigs/blank.png';
});

function showROITip(evt, id, nspk, amp, fdhm, tau, d2m) {
  const tip = document.getElementById('roi-tooltip');
  tip.innerHTML =
    '<b>ROI&nbsp;' + id + '</b><br>' +
    'Sparks:&nbsp;' + nspk + '<br>' +
    'AMP:&nbsp;' + amp.toFixed(2) + '<br>' +
    'FDHM:&nbsp;' + fdhm.toFixed(1) + '&nbsp;ms<br>' +
    '&tau;:&nbsp;' + tau.toFixed(1) + '&nbsp;ms<br>' +
    'd₂mem:&nbsp;' + d2m.toFixed(2) + '&nbsp;µm';
  tip.style.display = 'block';
  _moveROITip(evt);
}
function _moveROITip(evt) {
  const tip = document.getElementById('roi-tooltip');
  const wrap = document.getElementById('roi-map-wrap');
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  let x = evt.clientX - rect.left + 10;
  let y = evt.clientY - rect.top - 10;
  if (x + 140 > rect.width)  x = evt.clientX - rect.left - 150;
  if (y < 0) y = 4;
  tip.style.left = x + 'px';
  tip.style.top  = y + 'px';
}
function hideROITip() {
  document.getElementById('roi-tooltip').style.display = 'none';
}

function setFrame(n) {
  cur = Math.max(1, Math.min(totalFrames, n));
  const p = String(cur).padStart(5, '0');
  document.getElementById('frame-img').src = './detFilm/F1' + p + '.png';
  document.getElementById('frame-counter').textContent = cur + ' / ' + totalFrames;
  document.getElementById('frame-label').textContent =
    'Detection Film — Frame ' + cur + ' / ' + totalFrames;
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') setFrame(cur + 1);
  if (e.key === 'ArrowLeft')  setFrame(cur - 1);
});

function sortTable(tblId, col) {
  const tbl = document.getElementById(tblId);
  const tbody = tbl.tBodies[0];
  const rows = Array.from(tbody.rows);
  const ths = tbl.tHead.rows[0].cells;
  const th = ths[col];
  const asc = !th.classList.contains('sorted-asc');
  for (let h of ths) { h.classList.remove('sorted-asc', 'sorted-desc'); }
  th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
  rows.sort((a, b) => {
    const va = a.cells[col]?.dataset?.val ?? a.cells[col]?.textContent ?? '';
    const vb = b.cells[col]?.dataset?.val ?? b.cells[col]?.textContent ?? '';
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  rows.forEach(r => tbody.appendChild(r));
}
"""


# ── Table builders ──────────────────────────────────────────────────────────

def _th(label: str, col: int, tbl_id: str) -> str:
    return (f'<th onclick="sortTable(\'{tbl_id}\',{col})">'
            f'{_esc(label)}</th>')


def _accepted_table(sparks: list[Any]) -> str:
    tid = "tbl-accepted"
    cols = ["ID", "X", "Y", "T", "d2mem", "d2spk", "AMP", "amp", "BL",
            "RoR", "mass", "t2p", "FDHM", "FWHM", "tau", "R²", "ROI"]
    header = "".join(_th(c, i, tid) for i, c in enumerate(cols))
    rows = []
    for s in sparks:
        sid = int(_get(s, "id", 0))
        fwhm = _get(s, "fwhm") or _get(s, "FWHM", 0.0)
        fdhm = _get(s, "fdhm") or _get(s, "FDHM", 0.0)
        bl   = _get(s, "BL")   or _get(s, "bl", 0.0)
        d2spk = float(_get(s, "dist2spk", 0.0) or 0.0)
        roi_id = int(_get(s, "roi", 0) or 0)
        vals = [
            None,  # button placeholder
            int(_get(s, "px", 0)),
            int(_get(s, "py", 0)),
            int(_get(s, "pt", 0)),
            float(_get(s, "dist2memb", 0.0) or 0.0),
            d2spk,
            float(_get(s, "AMP", 0.0) or 0.0),
            float(_get(s, "amp", 0.0) or 0.0),
            float(bl),
            float(_get(s, "ror", 0.0) or 0.0),
            float(_get(s, "mass", 0.0) or 0.0),
            float(_get(s, "t2p", 0.0) or 0.0),
            float(fdhm),
            float(fwhm),
            float(_get(s, "tau", 0.0) or 0.0),
            float(_get(s, "r2", 0.0) or 0.0),
            roi_id,
        ]
        btn = f'<button class="spk-btn" onclick="selectSpark({sid})">{sid}</button>'
        tds = [f'<td>{btn}</td>']
        for v in vals[1:]:
            sv = _fmt(v, 2) if isinstance(v, float) else str(v)
            tds.append(f'<td data-val="{sv}">{sv}</td>')
        rows.append(f'<tr id="spkrow-{sid}">{"".join(tds)}</tr>')
    return (f'<table id="{tid}"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _roi_table(rois: list[Any]) -> str:
    tid = "tbl-rois"
    cols = ["ID", "X", "Y", "d2mem", "d2close", "AMP", "amp", "BL",
            "RoR", "mass", "t2p", "FDHM", "FWHM", "tau", "#spk"]
    header = "".join(_th(c, i, tid) for i, c in enumerate(cols))
    rows = []
    for r in rois:
        btn = f'<button class="roi-btn" onclick="selectROI({r.id})">{r.id}</button>'
        floats = [r.dist2memb, r.dist2closest, r.AMP, r.amp, r.BL,
                  r.ror, r.mass, r.t2p, r.FDHM, r.FWHM, r.tau]
        tds = [f'<td>{btn}</td>',
               f'<td>{r.px}</td>',
               f'<td>{r.py}</td>']
        for v in floats:
            sv = _fmt(v)
            tds.append(f'<td data-val="{sv}">{sv}</td>')
        tds.append(f'<td>{r.Nspk}</td>')
        rows.append(f'<tr id="roirow-{r.id}">{"".join(tds)}</tr>')
    return (f'<table id="{tid}"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _rejected_table(sparks: list[Any]) -> str:
    tid = "tbl-rejected"
    cols = ["ID", "X", "Y", "T", "d2mem", "AMP", "amp", "BL",
            "RoR", "mass", "t2p", "FDHM", "FWHM", "tau", "R²", "Reason"]
    header = "".join(_th(c, i, tid) for i, c in enumerate(cols))
    rows = []
    for s in sparks:
        sid = int(_get(s, "id", 0))
        fwhm = _get(s, "fwhm") or _get(s, "FWHM", 0.0)
        fdhm = _get(s, "fdhm") or _get(s, "FDHM", 0.0)
        bl   = _get(s, "BL")   or _get(s, "bl", 0.0)
        reason = _esc(str(_get(s, "fail", "") or ""))
        vals = [
            int(_get(s, "px", 0)),
            int(_get(s, "py", 0)),
            int(_get(s, "pt", 0)),
            float(_get(s, "dist2memb", 0.0) or 0.0),
            float(_get(s, "AMP", 0.0) or 0.0),
            float(_get(s, "amp", 0.0) or 0.0),
            float(bl),
            float(_get(s, "ror", 0.0) or 0.0),
            float(_get(s, "mass", 0.0) or 0.0),
            float(_get(s, "t2p", 0.0) or 0.0),
            float(fdhm),
            float(fwhm),
            float(_get(s, "tau", 0.0) or 0.0),
            float(_get(s, "r2", 0.0) or 0.0),
        ]
        btn = f'<button class="spk-btn" onclick="selectSpark({sid})">{sid}</button>'
        tds = [f'<td>{btn}</td>']
        for v in vals:
            sv = _fmt(v, 2) if isinstance(v, float) else str(v)
            tds.append(f'<td data-val="{sv}">{sv}</td>')
        tds.append(f'<td>{reason}</td>')
        rows.append(f'<tr>{"".join(tds)}</tr>')
    return (f'<table id="{tid}"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


# ── Interactive ROI map ──────────────────────────────────────────────────────

_SUPERJET = [
    (0.00, (0,   0, 143)),
    (0.15, (0,   0, 255)),
    (0.35, (0, 255, 255)),
    (0.55, (255, 255,   0)),
    (0.75, (255, 128,   0)),
    (1.00, (255,   0,   0)),
]


def _roi_color(t: float) -> str:
    """Map t ∈ [0,1] to a superjet-like hex color."""
    t = max(0.0, min(1.0, t))
    for i in range(len(_SUPERJET) - 1):
        t0, c0 = _SUPERJET[i]
        t1, c1 = _SUPERJET[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ff0000"


def _roi_interactive_map(
    rois: list[Any],
    fo_shape: "tuple[int, int]",
    roi_r_px: float,
) -> str:
    """Return the HTML for the interactive ROI map section (SVG over Fo image)."""
    H, W = fo_shape
    r = roi_r_px
    font_size = f"{max(3.0, r * 0.85):.1f}"

    nspk_vals = [int(getattr(roi, "Nspk", 0) or 0) for roi in rois]
    nspk_min = min(nspk_vals) if nspk_vals else 0
    nspk_max = max(nspk_vals) if nspk_vals else 1
    nspk_range = max(1, nspk_max - nspk_min)

    rings: list[str] = []
    for roi in sorted(rois, key=lambda x: getattr(x, "id", 0)):
        rid   = getattr(roi, "id", 0)
        cx    = int(getattr(roi, "px", 0))
        cy    = int(getattr(roi, "py", 0))
        nspk  = int(getattr(roi, "Nspk", 0) or 0)
        amp   = float(getattr(roi, "AMP",       0.0) or 0.0)
        fdhm  = float(getattr(roi, "FDHM",      0.0) or 0.0)
        tau   = float(getattr(roi, "tau",        0.0) or 0.0)
        d2m   = float(getattr(roi, "dist2memb",  0.0) or 0.0)
        color = _roi_color((nspk - nspk_min) / nspk_range)
        rings.append(
            f'<g id="roi-marker-{rid}" class="roi-marker" style="--roi-color:{color}"'
            f' onclick="selectROI({rid})"'
            f' onmouseenter="showROITip(event,{rid},{nspk},{amp:.2f},{fdhm:.2f},{tau:.2f},{d2m:.2f})"'
            f' onmousemove="_moveROITip(event)"'
            f' onmouseleave="hideROITip()"'
            f' role="button" aria-label="ROI {rid}">'
            f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}"/>'
            f'<text x="{cx}" y="{cy}" font-size="{font_size}">{rid}</text>'
            f'</g>'
        )
    rings_svg = "\n      ".join(rings)

    # Vertical color legend — gradient from top (max) to bottom (min)
    v_stops = " ".join(
        f'<stop offset="{int(t * 100)}%" stop-color="{_roi_color(1.0 - t)}"/>'
        for t in (0.0, 0.25, 0.45, 0.65, 0.85, 1.0)
    )
    legend = (
        f'<div class="roi-legend-vert">'
        f'<svg xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="roi-grad-v" x1="0" x2="0" y1="0" y2="1">'
        f'{v_stops}</linearGradient></defs>'
        f'<rect x="5" y="12%" width="10" height="72%" rx="2" fill="url(#roi-grad-v)"/>'
        f'<text x="10" y="9%" font-size="6" fill="currentColor" text-anchor="middle">{nspk_max}</text>'
        f'<text x="10" y="96%" font-size="6" fill="currentColor" text-anchor="middle">{nspk_min}</text>'
        f'</svg></div>'
    )

    return f"""
    <div class="right-section">
      <h3>ROI Map</h3>
      <div class="roi-map-outer">
        <div id="roi-map-wrap" style="aspect-ratio: {W} / {H};">
          <img id="roi-map-bg" src="extraFigs/Fo_raw.png" alt="F₀ map">
          <svg id="roi-map-svg" viewBox="0 0 {W} {H}"
               preserveAspectRatio="xMidYMid meet"
               xmlns="http://www.w3.org/2000/svg">
        {rings_svg}
          </svg>
          <div id="roi-tooltip"></div>
        </div>
        {legend}
      </div>
    </div>"""


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_report(
    result: Phase10Result,
    output_dir: str | Path,
    title: str,
    dx_um: float,
    dt_ms: float,
    roi_r_um: float = 1.5,
    volume: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    fo_map: np.ndarray | None = None,
    generate_film: bool = True,
) -> Path:
    """Generate the HTML report and all supporting figures.

    Creates output_dir/extraFigs/, output_dir/spkFeat/, output_dir/roiFeat/.
    Writes output_dir/<title>.html.
    Returns the path to the HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("extraFigs", "spkFeat", "roiFeat"):
        (output_dir / sub).mkdir(exist_ok=True)

    sparks = result.clustering.sparks
    rois   = result.clustering.rois
    summary = result.summary

    accepted = [s for s in sparks if getattr(s, "good", True)]
    rejected = [s for s in sparks if not getattr(s, "good", True)]

    roi_y_values: list[np.ndarray] = []
    for r in rois:
        signal = getattr(r, "signal", None)
        if signal is not None and len(signal) > 0:
            arr = np.asarray(signal, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                roi_y_values.append(arr)
    for s in sparks:
        signal = _get(s, "timesignal")
        if signal is not None and len(signal) > 0:
            arr = np.asarray(signal, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                roi_y_values.append(arr)
    if roi_y_values:
        roi_all_y = np.concatenate(roi_y_values)
        roi_y_limits = (min(float(np.min(roi_all_y)), 0.0), float(np.max(roi_all_y)))
    else:
        roi_y_limits = None

    # ── Generate figures ────────────────────────────────────────────────────
    generate_blank(output_dir)
    if fo_map is not None:
        generate_fo_map(fo_map, output_dir)
        generate_fo_map_raw(fo_map, output_dir)
    if mask is not None:
        generate_mask_image(mask, output_dir)
    generate_location_sparks(accepted, mask, dx_um, output_dir, volume=volume)
    generate_location_sparks2(accepted, mask, output_dir, volume=volume)
    generate_location_rois(rois, mask, dx_um, output_dir,
                           fo_map=fo_map, volume=volume, roi_r_um=roi_r_um)
    if rois and fo_map is not None:
        generate_roi_heatmap(rois, fo_map.shape[:2], roi_r_um, dx_um, mask, output_dir)

    for s in sparks:
        generate_spark_feature_plot(s, volume, dx_um, dt_ms, output_dir, max_val=None)

    for r in rois:
        generate_roi_signal(r, sparks, dx_um, dt_ms, roi_r_um, output_dir, y_limits=roi_y_limits)

    if generate_film and volume is not None:
        generate_detection_film(volume, accepted, rejected, output_dir,
                                mask=mask, fo_map=fo_map)

    # Film frame count
    if volume is not None:
        film_total = volume.shape[2]
    else:
        film_total = max(1, int(round(summary.duration_s * 1000.0 / dt_ms)) + 1) if dt_ms > 0 else 1

    # ── KPI bar values ──────────────────────────────────────────────────────
    kpis = [
        ("Pixel (µm)", f"{dx_um:.3g}"),
        ("Frame (ms)", f"{dt_ms:.3g}"),
        ("Duration (s)", f"{summary.duration_s:.1f}"),
        ("Area (µm²)", f"{summary.cell_area_um2:.0f}"),
        ("Cell F₀", f"{summary.mean_fo:.1f}"),
        ("Candidates", str(summary.n_candidates)),
        ("Accepted", str(summary.n_accepted)),
        ("Rejected", str(summary.n_rejected)),
        ("Freq (spk/s)", f"{summary.spark_freq:.3f}"),
        ("Density (spk/s/mm²)", f"{summary.spark_density:.0f}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><span class="kpi-lbl">{_esc(k)}</span>'
        f'<span class="kpi-val">{_esc(v)}</span></div>'
        for k, v in kpis
    )

    # ── Tables ──────────────────────────────────────────────────────────────
    acc_tbl = _accepted_table(accepted)
    roi_tbl = _roi_table(rois)
    rej_tbl = _rejected_table(rejected)

    # ── Spark-to-frame map for JS (pt is 1-based frame index) ──────────────
    spark_frames_js = "{" + ", ".join(
        f"{int(_get(s, 'id', 0))}: {int(_get(s, 'pt', 1) or 1)}"
        for s in sparks
    ) + "}"

    spark_to_roi_js = "{" + ", ".join(
        f"{int(_get(s, 'id', 0))}: {int(_get(s, 'roi', 0) or 0)}"
        for s in sparks
        if int(_get(s, "roi", 0) or 0) > 0
    ) + "}"

    first_spark_by_roi: dict[int, int] = {}
    for s in sorted(accepted, key=lambda sp: int(_get(sp, "id", 0) or 0)):
        roi_id = int(_get(s, "roi", 0) or 0)
        sid = int(_get(s, "id", 0) or 0)
        if roi_id > 0 and sid > 0 and roi_id not in first_spark_by_roi:
            first_spark_by_roi[roi_id] = sid
    roi_to_first_spark_js = "{" + ", ".join(
        f"{roi_id}: {sid}" for roi_id, sid in sorted(first_spark_by_roi.items())
    ) + "}"

    # ── Inline JS with substituted values ───────────────────────────────────
    js = (_JS
          .replace("FILM_TOTAL", str(film_total))
          .replace("SPARK_FRAMES", spark_frames_js)
          .replace("SPARK_TO_ROI", spark_to_roi_js)
          .replace("ROI_TO_FIRST_SPARK", roi_to_first_spark_js))

    # ── Optional right-panel sections ───────────────────────────────────────
    # Interactive ROI map: SVG overlay on Fo background (requires fo_map + rois).
    # Falls back to the static LocationRois.png when fo_map is not available.
    # Static ROI Locations always shown
    roi_locations_section = """
    <div class="right-section">
      <h3>ROI Locations</h3>
      <img src="extraFigs/LocationRois.png" alt="ROI locations">
    </div>"""

    roi_heatmap_section = ""
    if rois and fo_map is not None:
        roi_heatmap_section = """
    <div class="right-section">
      <h3>ROI Heatmap</h3>
      <img src="extraFigs/LocationRoisHeat.png" alt="ROI heatmap">
    </div>"""

    if fo_map is not None and rois:
        roi_r_px = roi_r_um / dx_um if dx_um > 0 else 8.0
        roi_map_section = _roi_interactive_map(rois, fo_map.shape[:2], roi_r_px)
        fo_section = ""  # Fo already used as the map background
    else:
        roi_map_section = ""
        fo_section = ""
        if fo_map is not None:
            fo_section = """
    <div class="right-section">
      <h3>F₀ Map</h3>
      <img src="extraFigs/Fo.png" alt="F0 map">
    </div>"""

    mask_section = ""
    if mask is not None:
        mask_section = """
    <div class="right-section">
      <h3>Cell Mask</h3>
      <img src="extraFigs/mask.png" alt="Cell mask">
    </div>"""

    # ── Assemble HTML ────────────────────────────────────────────────────────
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>

<!-- ── Header ── -->
<div id="hdr">
  <h1>{_esc(title)}</h1>
  <div class="kpi-bar">{kpi_html}</div>
</div>

<!-- ── ROI signal strip (full-width, shown on ROI click, hidden by default) ── -->
<div id="roi-strip">
  <img id="roi-strip-img" src="extraFigs/blank.png" alt="ROI signal — double-click to close">
</div>

<!-- ── Main layout ── -->
<div id="main">

  <!-- Left: tables -->
  <div class="col-left">

    <div class="tbl-panel panel-green">
      <div class="tbl-hd">
        &#10003; Accepted sparks
        <span class="count">({len(accepted)})</span>
      </div>
      <div class="tbl-wrap">{acc_tbl}</div>
    </div>

    <div class="tbl-panel panel-blue">
      <div class="tbl-hd">
        &#9670; ROIs
        <span class="count">({len(rois)})</span>
      </div>
      <div class="tbl-wrap">{roi_tbl}</div>
    </div>

    <div class="tbl-panel panel-red">
      <div class="tbl-hd">
        &#10007; Rejected sparks
        <span class="count">({len(rejected)})</span>
      </div>
      <div class="tbl-wrap">{rej_tbl}</div>
    </div>

  </div><!-- /col-left -->

  <!-- Centre: spark images -->
  <div class="col-centre">
    <div class="centre-title" id="ctr-title">Select a spark &#8594;</div>
    <div class="centre-pair">
      <img id="ctr-feat" src="extraFigs/blank.png" alt="Spark feature plot">
    </div>
  </div><!-- /col-centre -->

  <!-- Right: maps + film -->
  <div class="col-right">

{roi_map_section}{roi_locations_section}{roi_heatmap_section}
    <div class="right-section">
      <h3>Spark Locations</h3>
      <img src="extraFigs/LocationSparks.png" alt="Spark locations">
    </div>

    <div class="right-section">
      <h3>Spark Labels</h3>
      <img src="extraFigs/LocationSparks2.png" alt="Spark labels">
    </div>

{fo_section}{mask_section}
    <div class="right-section">
      <h3 id="frame-label">Detection Film &#8212; Frame 1 / {film_total}</h3>
      <div id="film-ctrl">
        <button onclick="setFrame(cur-1)">&#8592;</button>
        <span id="frame-counter">1 / {film_total}</span>
        <button onclick="setFrame(cur+1)">&#8594;</button>
      </div>
      <img id="frame-img"
           src="./detFilm/F100001.png"
           style="width:100%;display:block;"
           alt="Detection film frame"/>
    </div>

  </div><!-- /col-right -->

</div><!-- /main -->

<script>
{js}
</script>
</body>
</html>"""

    out_path = output_dir / f"{title}.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path
