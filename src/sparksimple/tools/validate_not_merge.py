"""Browser-based manual validation for existing SparkSimple2 results.

This is the Python port of ``mfiles/ValidateNotMerge.m``.  It opens a local
Flask UI, shows one spark at a time, records manual accept/reject decisions,
then rewrites the result exports with updated ``spkF.good`` values.
"""

from __future__ import annotations

import html
import os
import re
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from sparksimple.export.matlab_compat import save_zdata2_mat
from sparksimple.export.models import ExportPaths, ExportSummary
from sparksimple.export.pipeline import Phase10Result
from sparksimple.export.tables import generate_accepted_csv, generate_rejected_csv
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.report.html import _CSS as REPORT_CSS
from sparksimple.report.html import generate_report
from sparksimple.roi.cluster import cluster_rois


RF = "iT_03_r2_03"
PORT = 5555


_CSS = REPORT_CSS + """
:root {
  --surface-2: #0f141b;
  --ok: #3fb950;
  --bad: #f85149;
}
header {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 14px;
  min-height: 58px;
  padding: 10px 16px;
  background: var(--surface);
  border-bottom: 2px solid var(--accent);
}
h1 {
  margin: 0;
  color: var(--accent);
  font-size: 17px;
  font-weight: 700;
  flex: 1 1 auto;
}
.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 80px;
  font-variant-numeric: tabular-nums;
}
.stat span:first-child {
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
}
.stat span:last-child { color: var(--accent); font-weight: 700; }
.actions { display: flex; gap: 8px; }
button {
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 12px;
  color: var(--text);
  background: var(--surface-2);
  cursor: pointer;
  font-weight: 700;
}
button:hover { border-color: var(--accent); }
button.accept { background: #12261a; color: var(--ok); }
button.reject { background: #2d1111; color: var(--bad); }
main {
  height: calc(100vh - 58px);
  display: grid;
  grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
  grid-template-rows: 1fr auto;
  gap: 12px;
  padding: 12px;
}
.panel {
  min-height: 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  display: flex;
  flex-direction: column;
}
.panel h2 {
  margin: 0;
  padding: 9px 12px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
}
.media {
  min-height: 0;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
  background: #05070a;
}
.media img {
  display: block;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.film-controls {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 8px;
  border-top: 1px solid var(--border);
}
#frame-counter { min-width: 110px; color: var(--muted); text-align: center; }
.hints {
  grid-column: 1 / -1;
  color: var(--muted);
  padding: 4px 2px;
}
.finished {
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
}
.finished h1 { flex: initial; font-size: 28px; }
@media (max-width: 800px) {
  header { align-items: flex-start; flex-wrap: wrap; }
  main { grid-template-columns: 1fr; grid-template-rows: minmax(260px, 1fr) minmax(260px, 1fr) auto; }
  .hints { grid-column: auto; }
}
"""


@dataclass
class ValidationContext:
    experiment_dir: Path
    result_dir: Path
    title: str
    sparks: list[FilteredSpark]
    film_frames: list[Path]
    feature_plots: dict[int, Path]
    decisions: dict[int, bool] = field(default_factory=dict)
    cursor: int = 0
    processing: bool = False
    done: bool = False
    error: str | None = None
    report_path: Path | None = None

    @property
    def total(self) -> int:
        return len(self.sparks)

    @property
    def accepted_count(self) -> int:
        return sum(1 for v in self.decisions.values() if v)

    @property
    def rejected_count(self) -> int:
        return sum(1 for v in self.decisions.values() if not v)

    @property
    def current_spark(self) -> FilteredSpark | None:
        if self.cursor >= self.total:
            return None
        return self.sparks[self.cursor]


class _ServerThread(threading.Thread):
    def __init__(self, app: Any, port: int) -> None:
        from werkzeug.serving import make_server

        super().__init__(daemon=True)
        self.server = make_server("127.0.0.1", port, app)
        app.app_context().push()

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


def _field(record: Any, name: str, default: Any = 0) -> Any:
    try:
        value = record[name]
    except Exception:
        value = getattr(record, name, default)
    return _unwrap(value)


def _unwrap(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype.names is None:
        return value.reshape(-1)[0].item()
    return value


def _as_float(record: Any, *names: str, default: float = 0.0) -> float:
    for name in names:
        value = _field(record, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return default


def _as_int(record: Any, name: str, default: int = 0) -> int:
    try:
        return int(round(float(_field(record, name, default))))
    except (TypeError, ValueError):
        return default


def _as_array(record: Any, name: str) -> np.ndarray:
    value = _field(record, name, np.array([]))
    if isinstance(value, np.ndarray):
        return value
    if value is None:
        return np.array([])
    return np.asarray(value)


def _as_text(record: Any, name: str, default: str = "") -> str:
    value = _field(record, name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(v) for v in value.reshape(-1)).strip()
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _normalise_spkf(spkf: Any) -> list[Any]:
    arr = np.asarray(spkf)
    if arr.dtype.names:
        return [arr.reshape(-1)[i] for i in range(arr.size)]
    return list(arr.reshape(-1))


def load_sparks_from_zdata2(zdata2_path: str | Path) -> list[FilteredSpark]:
    mat = scipy.io.loadmat(str(zdata2_path), squeeze_me=True, struct_as_record=False)
    if "spkF" not in mat:
        raise KeyError(f"Missing spkF in {zdata2_path}")

    sparks: list[FilteredSpark] = []
    for i, rec in enumerate(_normalise_spkf(mat["spkF"]), start=1):
        px = _as_int(rec, "px")
        py = _as_int(rec, "py")
        pt = _as_int(rec, "pt")
        feature = SparkFeature(
            px=px,
            py=py,
            pt=pt,
            x_0b=px - 1,
            y_0b=py - 1,
            t_0b=pt - 1,
            dist2memb=_as_float(rec, "dist2memb"),
            fwhm=_as_float(rec, "fwhm", "FWHM"),
            timesignal=_as_array(rec, "timesignal").reshape(-1),
            timeinterval=_as_array(rec, "timeinterval").reshape(-1),
            r2=_as_float(rec, "r2"),
            amp=_as_float(rec, "amp"),
            tau=_as_float(rec, "tau"),
            ror=_as_float(rec, "ror"),
            t2p=_as_float(rec, "t2p"),
            fdhm=_as_float(rec, "fdhm", "FDHM"),
            AMP=_as_float(rec, "AMP"),
            BL=_as_float(rec, "BL", "bl"),
            mass=_as_float(rec, "mass"),
        )
        setattr(feature, "dist2spk", _as_float(rec, "dist2spk"))
        setattr(feature, "roi", _as_int(rec, "roi"))
        spark_id = _as_int(rec, "id", i) or i
        good = bool(_as_int(rec, "good", 1))
        fail = _as_text(rec, "fail", "no" if good else "manual rejection") or "no"
        sparks.append(FilteredSpark(id=spark_id, good=good, fail=fail, feature=feature))
    return sparks


def _load_metadata(result_dir: Path) -> dict[str, Any]:
    meta_path = result_dir / "zMetaData.mat"
    if meta_path.exists():
        return scipy.io.loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)
    return {}


def _meta_value(meta: dict[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name in meta:
            return _unwrap(meta[name])
    return default


def _required_positive_metadata(meta: dict[str, Any], *names: str) -> float:
    """Return the first present field in *names* as a finite, positive float.

    SparkSimple never invents spatial or temporal calibration: DX and DT must
    be present, numeric, finite and positive, or the tool aborts before it
    touches any output.
    """
    for name in names:
        if name in meta:
            raw = _unwrap(meta[name])
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Mandatory acquisition metadata {name!r} is not numeric: {raw!r}"
                ) from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Mandatory acquisition metadata {name!r} must be finite and "
                    f"positive; got {value!r}"
                )
            return value
    raise KeyError(f"Missing mandatory acquisition metadata: {'/'.join(names)}")


def _load_volume_and_mask(result_dir: Path, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, bool]:
    mask = np.asarray(_meta_value(meta, "mask", "MASK", default=np.ones((1, 1))), dtype=float)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    temps = int(round(float(_meta_value(meta, "temps", "T", default=1)))) or 1

    for mat_name in ("zData0.mat", "zData1.mat", "zMetaData.mat"):
        path = result_dir / mat_name
        if not path.exists():
            continue
        data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        for key in ("volum", "volume", "volum2", "dFF", "F"):
            if key in data:
                volume = np.asarray(data[key], dtype=float)
                if volume.ndim == 3:
                    return volume, mask, True

    if mask.ndim != 2:
        mask = np.ones((1, 1), dtype=float)
    return np.zeros((mask.shape[0], mask.shape[1], temps), dtype=float), mask, False


def _extract_title(meta: dict[str, Any], experiment_dir: Path, result_dir: Path) -> str:
    for name in ("title", "Title", "titol"):
        if name in meta:
            value = _unwrap(meta[name])
            if value is not None:
                text = str(value).strip()
                if text:
                    return re.sub(r"[^\w.-]+", "_", text)
    return re.sub(r"[^\w.-]+", "_", experiment_dir.name or result_dir.name)


def _write_validation_mat(result_dir: Path, decisions: dict[int, bool]) -> None:
    lines = np.empty((len(decisions),), dtype=object)
    for idx, spark_id in enumerate(sorted(decisions)):
        lines[idx] = f"    Spark {spark_id} : {1 if decisions[spark_id] else 0}"
    scipy.io.savemat(str(result_dir / "zValidation.mat"), {"parrafada": lines})


def apply_manual_validation(
    result_dir: str | Path,
    decisions: dict[int, bool],
    *,
    title: str | None = None,
    report: bool = True,
    roi_r_um: float | None = None,
) -> Phase10Result:
    """Apply manual decisions and regenerate validation outputs.

    Args:
        result_dir: Result folder containing ``zData2.mat`` and ``zMetaData.mat``.
        decisions: Mapping of 1-based spark id to accepted/rejected value.
        title: Optional title for CSV/HTML filenames.
        report: Whether to regenerate the HTML report.
        roi_r_um: Optional override for ROI clustering radius.
    """
    result_path = Path(result_dir)
    meta = _load_metadata(result_path)
    sparks = load_sparks_from_zdata2(result_path / "zData2.mat")

    for spark in sparks:
        if spark.id in decisions:
            accepted = bool(decisions[spark.id])
            spark.good = accepted
            spark.fail = "no" if accepted else (
                "manual rejection" if spark.fail == "no" else spark.fail
            )

    volume, mask, has_real_volume = _load_volume_and_mask(result_path, meta)
    dx_um = _required_positive_metadata(meta, "DX", "dx_um")
    dt_ms = _required_positive_metadata(meta, "DT", "dt_ms")
    roi_radius = float(roi_r_um if roi_r_um is not None else _meta_value(meta, "roiR", default=2.0))
    temps = int(volume.shape[2])
    fo_map = _meta_value(meta, "fo", "FO", "fo_map", default=1.0)
    if isinstance(fo_map, np.ndarray):
        fo_map = np.asarray(fo_map, dtype=float)
    else:
        fo_map = float(fo_map)
    out_title = title or _extract_title(meta, result_path.parent, result_path)

    clustering = cluster_rois(sparks, volume, mask, dx_um, dt_ms, roi_radius)
    rois = clustering.rois

    accepted_csv = result_path / f"yData_{out_title}.csv"
    rejected_csv = result_path / f"zDataRejected_{out_title}.csv"
    zdata2 = result_path / "zData2.mat"
    zmetadata = result_path / "zMetaData.mat"

    generate_accepted_csv(sparks, rois, fo_map, mask, dx_um, dt_ms, temps, str(accepted_csv))
    generate_rejected_csv(sparks, fo_map, mask, dx_um, dt_ms, temps, str(rejected_csv))
    save_zdata2_mat(str(zdata2), sparks, rois)
    _write_validation_mat(result_path, decisions)

    duration_s = max(0, temps - 1) * dt_ms / 1000.0
    cell_area_um2 = float(np.sum(mask == 1) * dx_um * dx_um)
    n_accepted = sum(1 for s in sparks if s.good)
    if isinstance(fo_map, np.ndarray):
        fo_in_mask = fo_map[mask == 1]
        mean_fo = float(np.mean(fo_in_mask)) if fo_in_mask.size else float(np.mean(fo_map))
        std_fo = float(np.std(fo_in_mask, ddof=1)) if fo_in_mask.size > 1 else 0.0
    else:
        mean_fo = float(fo_map)
        std_fo = 0.0
    spark_density_per_um2_s = (
        n_accepted / duration_s / cell_area_um2
        if duration_s > 0 and cell_area_um2 > 0 else 0.0
    )
    spark_density_per_mm2_s = spark_density_per_um2_s * 1e6
    summary = ExportSummary(
        n_candidates=len(sparks),
        n_accepted=n_accepted,
        n_rejected=len(sparks) - n_accepted,
        spark_freq=n_accepted / duration_s if duration_s > 0 else 0.0,
        spark_density=spark_density_per_mm2_s,
        spark_density_per_um2_s=spark_density_per_um2_s,
        spark_density_per_mm2_s=spark_density_per_mm2_s,
        cell_area_um2=cell_area_um2,
        duration_s=duration_s,
        mean_fo=mean_fo,
        std_fo=std_fo,
    )
    paths = ExportPaths(
        accepted_csv=str(accepted_csv),
        rejected_csv=str(rejected_csv),
        zdata2_mat=str(zdata2),
        zmetadata_mat=str(zmetadata),
    )
    result = Phase10Result(clustering=clustering, summary=summary, paths=paths)
    if report:
        html_path = generate_report(
            result=result,
            output_dir=result_path,
            title=out_title,
            dx_um=dx_um,
            dt_ms=dt_ms,
            roi_r_um=roi_radius,
            volume=volume if has_real_volume else None,
            mask=mask,
            fo_map=fo_map if isinstance(fo_map, np.ndarray) else None,
            generate_film=False,
        )
        result.paths.html_report = str(html_path)
    return result


def _select_experiment_folder() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select SparkSimple experiment folder")
    root.destroy()
    if not folder:
        raise RuntimeError("No experiment folder selected.")
    return Path(folder)


def _find_result_dir(experiment_dir: Path, rf_tag: str) -> Path:
    exact = experiment_dir / rf_tag
    if exact.is_dir():
        return exact
    matches = sorted(p for p in experiment_dir.glob(f"{rf_tag}*") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"No result folder matching '{rf_tag}' in {experiment_dir}")
    return matches[0]


def _collect_frames(result_dir: Path) -> list[Path]:
    frames = sorted((result_dir / "detFilm").glob("F1*.png"))
    if not frames:
        frames = sorted((result_dir / "detFilm").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {result_dir / 'detFilm'}")
    return frames


def _collect_feature_plots(result_dir: Path, sparks: list[FilteredSpark]) -> dict[int, Path]:
    plots: dict[int, Path] = {}
    for spark in sparks:
        path = result_dir / "spkFeat" / f"Spk{spark.id}.png"
        if path.exists():
            plots[spark.id] = path
    return plots


def build_context(experiment_dir: Path, rf_tag: str = RF) -> ValidationContext:
    result_dir = _find_result_dir(experiment_dir, rf_tag)
    sparks = load_sparks_from_zdata2(result_dir / "zData2.mat")
    meta = _load_metadata(result_dir)
    title = _extract_title(meta, experiment_dir, result_dir)
    return ValidationContext(
        experiment_dir=experiment_dir,
        result_dir=result_dir,
        title=title,
        sparks=sparks,
        film_frames=_collect_frames(result_dir),
        feature_plots=_collect_feature_plots(result_dir, sparks),
    )


def _image_route(ctx: ValidationContext, kind: str, filename: str):
    from flask import send_from_directory

    roots = {
        "film": ctx.result_dir / "detFilm",
        "feature": ctx.result_dir / "spkFeat",
        "extra": ctx.result_dir / "extraFigs",
    }
    root = roots.get(kind)
    if root is None:
        return ("Unknown image kind", 404)
    return send_from_directory(root, filename)


def _spark_payload(ctx: ValidationContext) -> dict[str, Any]:
    from flask import url_for

    spark = ctx.current_spark
    if spark is None:
        return {"finished": True}
    frame_index = max(1, min(len(ctx.film_frames), int(spark.feature.pt)))
    return {
        "finished": False,
        "sparkId": spark.id,
        "index": ctx.cursor + 1,
        "total": ctx.total,
        "pt": int(spark.feature.pt),
        "accepted": ctx.accepted_count,
        "rejected": ctx.rejected_count,
        "frameIndex": frame_index,
        "frameUrl": url_for("image", kind="film", filename=ctx.film_frames[frame_index - 1].name),
        "featureUrl": url_for("image", kind="feature", filename=f"Spk{spark.id}.png"),
        "hasFeature": spark.id in ctx.feature_plots,
    }


def create_app(ctx: ValidationContext, *, shutdown_callback: Any = None) -> Any:
    from flask import Flask, jsonify, redirect, request, url_for

    app = Flask(__name__)

    @app.get("/")
    def index():
        if ctx.done:
            return redirect(url_for("done"))
        return _render_index(ctx)

    @app.get("/state")
    def state():
        payload = _spark_payload(ctx)
        payload.update({
            "processing": ctx.processing,
            "done": ctx.done,
            "error": ctx.error,
            "report": str(ctx.report_path) if ctx.report_path else None,
        })
        return jsonify(payload)

    @app.post("/validate")
    def validate():
        data = request.get_json(force=True) or {}
        spark_id = int(data.get("spark_id", 0))
        accepted = bool(data.get("accepted"))
        spark = ctx.current_spark
        if spark is None or spark.id != spark_id:
            return jsonify({"error": "Spark is no longer current."}), 409
        ctx.decisions[spark_id] = accepted
        ctx.cursor += 1
        if ctx.cursor >= ctx.total and not ctx.processing:
            ctx.processing = True
            thread = threading.Thread(target=_process_finished, args=(ctx,), daemon=True)
            thread.start()
        return jsonify(_spark_payload(ctx))

    @app.get("/done")
    def done():
        if not ctx.done:
            return _render_processing(ctx)
        if shutdown_callback is not None:
            threading.Timer(1.0, shutdown_callback).start()
        return _render_done(ctx)

    @app.get("/image/<kind>/<path:filename>")
    def image(kind: str, filename: str):
        return _image_route(ctx, kind, filename)

    return app


def _process_finished(ctx: ValidationContext) -> None:
    try:
        result = apply_manual_validation(ctx.result_dir, ctx.decisions, title=ctx.title, report=True)
        ctx.report_path = Path(result.paths.html_report) if result.paths.html_report else None
        ctx.done = True
    except Exception as exc:
        ctx.error = str(exc)
    finally:
        ctx.processing = False


def _render_index(ctx: ValidationContext) -> str:
    from flask import url_for

    state = _spark_payload(ctx)
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validate {html.escape(ctx.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1 id="title">Spark {state['index']} / {state['total']} - Experiment: {html.escape(ctx.title)}</h1>
  <div class="stat"><span>Accepted</span><span id="accepted">{state['accepted']}</span></div>
  <div class="stat"><span>Rejected</span><span id="rejected">{state['rejected']}</span></div>
  <div class="actions">
    <button class="accept" onclick="validateSpark(true)">ACCEPT</button>
    <button class="reject" onclick="validateSpark(false)">REJECT</button>
    <button id="theme-btn" onclick="toggleTheme()">☀ Light</button>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Film strip</h2>
    <div class="media"><img id="film" src="{state['frameUrl']}" alt="Detection film frame"></div>
    <div class="film-controls">
      <button onclick="moveFrame(-1)">&#8592;</button>
      <span id="frame-counter">{state['frameIndex']} / {len(ctx.film_frames)}</span>
      <button onclick="moveFrame(1)">&#8594;</button>
    </div>
  </section>
  <section class="panel">
    <h2>Feature plot</h2>
    <div class="media"><img id="feature" src="{state['featureUrl']}" alt="Spark feature plot"></div>
  </section>
  <div class="hints">Keyboard: &#8592; &#8594; film, Y/N accept/reject, Space = confirm selected action.</div>
</main>
<script>
const frames = {[
    url_for("image", kind="film", filename=frame.name) for frame in ctx.film_frames
]!r};
let current = {state['frameIndex']};
let spark = {state['sparkId']};
let sparkIndex = {state['index']};
let total = {state['total']};
let pending = null;
let busy = false;
function toggleTheme() {{
  const root = document.documentElement;
  const theme = root.getAttribute('data-theme');
  root.setAttribute('data-theme', theme === 'light' ? 'dark' : 'light');
  document.getElementById('theme-btn').textContent =
    theme === 'light' ? '☀ Light' : '🌙 Dark';
}}
function moveFrame(delta) {{
  current = Math.max(1, Math.min(frames.length, current + delta));
  document.getElementById('film').src = frames[current - 1];
  document.getElementById('frame-counter').textContent = current + ' / ' + frames.length;
}}
async function validateSpark(accepted) {{
  if (busy) return;
  busy = true;
  pending = accepted;
  const res = await fetch('/validate', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{spark_id: spark, accepted}})
  }});
  const data = await res.json();
  if (!res.ok) {{
    alert(data.error || 'Validation failed');
    busy = false;
    return;
  }}
  if (data.finished) {{
    document.body.innerHTML = `{_processing_body(ctx)}`;
    waitDone();
    return;
  }}
  pending = null;
  spark = data.sparkId;
  sparkIndex = data.index;
  current = data.frameIndex;
  document.getElementById('title').textContent =
    'Spark ' + data.index + ' / ' + data.total + ' - Experiment: {html.escape(ctx.title)}';
  document.getElementById('accepted').textContent = data.accepted;
  document.getElementById('rejected').textContent = data.rejected;
  document.getElementById('film').src = data.frameUrl;
  document.getElementById('feature').src = data.featureUrl;
  document.getElementById('frame-counter').textContent = current + ' / ' + frames.length;
  busy = false;
}}
async function waitDone() {{
  const res = await fetch('/state');
  const data = await res.json();
  if (data.error) {{
    document.body.innerHTML = `<div class="finished"><h1>Processing failed</h1><p>${{data.error}}</p></div>`;
    return;
  }}
  if (data.done) {{
    window.location = '/done';
    return;
  }}
  setTimeout(waitDone, 1000);
}}
document.addEventListener('keydown', (event) => {{
  if (event.key === 'ArrowLeft' || event.key === '4') moveFrame(-1);
  if (event.key === 'ArrowRight' || event.key === '6') moveFrame(1);
  if (event.key.toLowerCase() === 'y') validateSpark(true);
  if (event.key.toLowerCase() === 'n') validateSpark(false);
  if (event.code === 'Space' && pending !== null) validateSpark(pending);
}});
</script>
</body>
</html>"""


def _processing_body(ctx: ValidationContext) -> str:
    return (
        '<div class="finished">'
        '<h1>Finished - processing...</h1>'
        f"<p>{html.escape(ctx.result_dir.name)}</p>"
        "</div>"
    )


def _render_processing(ctx: ValidationContext) -> str:
    return f'<!doctype html><html data-theme="dark"><head><style>{_CSS}</style></head><body>{_processing_body(ctx)}</body></html>'


def _render_done(ctx: ValidationContext) -> str:
    report = f"<p>Report: {html.escape(str(ctx.report_path))}</p>" if ctx.report_path else ""
    return f"""<!doctype html>
<html data-theme="dark"><head><style>{_CSS}</style></head>
<body><div class="finished">
<h1>Done</h1>
<p>Validation saved for {html.escape(ctx.result_dir.name)}.</p>
{report}
<p>This local server will stop automatically.</p>
</div></body></html>"""


def run_server(experiment_dir: Path | None = None, *, rf_tag: str = RF, port: int = PORT) -> None:
    folder = experiment_dir or _select_experiment_folder()
    ctx = build_context(Path(folder), rf_tag)
    holder: dict[str, _ServerThread] = {}
    app = create_app(ctx, shutdown_callback=lambda: holder["server"].shutdown())
    server = _ServerThread(app, port)
    holder["server"] = server
    server.start()
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    try:
        while server.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        server.shutdown()


def main() -> None:
    run_server()
