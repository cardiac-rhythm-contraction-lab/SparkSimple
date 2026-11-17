"""Browser UI for manual correction of spark distance-to-membrane values."""

from __future__ import annotations

import argparse
import base64
import math
import os
import struct
import threading
import time
import webbrowser
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from scipy import ndimage

from sparksimple.report.html import _CSS as REPORT_CSS


PORT = 5556


@dataclass
class SparkRef:
    """Reference to a MATLAB struct element in the loaded spkF/SPARKS array."""

    index: tuple[int, ...]
    number: int
    spark_id: int
    px: float
    py: float
    dist2memb: float


def compute_dist2memb(
    xs: float,
    ys: float,
    px: float,
    py: float,
    dx_um: float,
) -> float:
    """Return membrane distance in um from a clicked image coordinate."""
    return float(math.hypot(xs - px, ys - py) * dx_um)


def _mat_scalar(value: Any, default: Any = 0) -> Any:
    """Extract a scalar from scipy.io.loadmat MATLAB field values."""
    cur = value
    while isinstance(cur, np.ndarray):
        if cur.size == 0:
            return default
        cur = cur.flat[0]
    try:
        return cur.item()
    except AttributeError:
        return cur


def _required_positive_dx(raw: Any) -> float:
    """Extract DX from zMetaData.mat with no fallback.

    Membrane-distance measurements are in micrometres, so an empty, non-numeric,
    non-finite, zero, or negative DX must abort the tool instead of silently
    becoming 1.0 px == 1.0 um.
    """
    value = _mat_scalar(raw, default=None)
    if value is None:
        raise ValueError("zMetaData.mat DX is empty; cannot derive membrane distances")
    try:
        dx_um = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"zMetaData.mat DX is not numeric: {value!r}") from exc
    if not math.isfinite(dx_um) or dx_um <= 0:
        raise ValueError(f"zMetaData.mat DX must be finite and positive; got {dx_um!r}")
    return dx_um


def _get_field(record: np.void, field: str, default: Any = 0) -> Any:
    if record.dtype.names is None or field not in record.dtype.names:
        return default
    return _mat_scalar(record[field], default)


def _set_field(record: np.void, field: str, value: Any) -> None:
    record[field] = np.array([[value]])


def _clean_mat_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if not k.startswith("__")}


def _normalise_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (np.nan_to_num(arr, nan=lo) - lo) / (hi - lo)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as a PNG using only the standard library."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("PNG encoder expects an RGB uint8 array")

    height, width, _ = rgb.shape

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack("!I", len(payload))
            + tag
            + payload
            + struct.pack("!I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _data_url(rgb: np.ndarray) -> str:
    encoded = base64.b64encode(_png_bytes(rgb)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _mask_contour(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if binary.ndim > 2:
        binary = np.squeeze(binary)
    eroded = ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool))
    return binary & ~eroded


def _draw_crosshair(rgb: np.ndarray, px: float, py: float) -> None:
    height, width, _ = rgb.shape
    x = int(round(px)) - 1
    y = int(round(py)) - 1
    if not (0 <= x < width and 0 <= y < height):
        return
    for yy in range(max(0, y - 1), min(height, y + 2)):
        rgb[yy, x] = (255, 255, 0)
    for xx in range(max(0, x - 1), min(width, x + 2)):
        rgb[y, xx] = (255, 255, 0)


def _find_result_dir() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on desktop runtime
        raise RuntimeError("Tkinter is required to choose a results folder") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        folder = filedialog.askdirectory(title="Select SparkSimple results folder")
    finally:
        root.destroy()
    if not folder:
        raise RuntimeError("No results folder selected")
    return Path(folder)


def _resolve_zdata0(result_dir: Path, is_merged: bool) -> Path:
    direct = result_dir / "zData0.mat"
    if direct.exists():
        return direct
    if is_merged:
        for candidate in sorted(result_dir.parent.glob("Results_new*")):
            zdata0 = candidate / "zData0.mat"
            if zdata0.exists():
                return zdata0
    raise FileNotFoundError(f"Could not find zData0.mat for {result_dir}")


class ModifyDist2MembraneState:
    """Loaded MATLAB data and browser-session state."""

    def __init__(self, result_dir: Path) -> None:
        self.result_dir = result_dir
        self.is_merged = (
            result_dir.name.upper() == "MERGED"
            or (result_dir / "zDataMERGE.mat").exists()
        )
        self.zdata_path = result_dir / ("zDataMERGE.mat" if self.is_merged else "zData2.mat")
        self.spark_var = "SPARKS" if self.is_merged else "spkF"
        self.meta_path = result_dir / "zMetaData.mat"
        self.zdata0_path = _resolve_zdata0(result_dir, self.is_merged)

        if not self.zdata_path.exists():
            raise FileNotFoundError(f"Could not find {self.zdata_path.name}")
        if not self.meta_path.exists():
            raise FileNotFoundError("Could not find zMetaData.mat")

        self.zdata = scipy.io.loadmat(self.zdata_path, squeeze_me=False)
        self.meta = scipy.io.loadmat(self.meta_path, squeeze_me=False)
        self.zdata0 = scipy.io.loadmat(self.zdata0_path, squeeze_me=False)

        if self.spark_var not in self.zdata:
            raise KeyError(f"{self.zdata_path.name} does not contain {self.spark_var}")
        if "DX" not in self.meta or "mask" not in self.meta:
            raise KeyError("zMetaData.mat must contain DX and mask")
        if "volum" not in self.zdata0:
            raise KeyError("zData0.mat must contain volum")

        self.dx_um = _required_positive_dx(self.meta["DX"])
        self.mask = np.asarray(self.meta["mask"])
        self.volume = np.asarray(self.zdata0["volum"])
        self.sparks = self.zdata[self.spark_var]
        self.accepted = self._accepted_sparks()
        self.current = 0
        self.finished = threading.Event()
        self.saved = False

        if not self.accepted:
            raise RuntimeError("No accepted sparks found (good == 1)")

        integrated = np.sum(self.volume, axis=2)
        green = _normalise_image(integrated)
        self.base_rgb = np.zeros((*green.shape, 3), dtype=np.uint8)
        self.base_rgb[:, :, 1] = green
        contour = _mask_contour(self.mask)
        self.base_rgb[contour] = (255, 0, 0)

    def _accepted_sparks(self) -> list[SparkRef]:
        accepted: list[SparkRef] = []
        for flat_idx in range(self.sparks.size):
            idx = np.unravel_index(flat_idx, self.sparks.shape)
            record = self.sparks[idx]
            if int(_get_field(record, "good", 0)) != 1:
                continue
            px = float(_get_field(record, "px", 0.0))
            py = float(_get_field(record, "py", 0.0))
            spark_id = int(_get_field(record, "id", flat_idx + 1))
            accepted.append(
                SparkRef(
                    index=idx,
                    number=flat_idx + 1,
                    spark_id=spark_id,
                    px=px,
                    py=py,
                    dist2memb=float(_get_field(record, "dist2memb", 0.0)),
                )
            )
        return accepted

    def current_spark(self) -> SparkRef | None:
        if self.current >= len(self.accepted):
            return None
        return self.accepted[self.current]

    def image_for_current(self) -> str:
        spark = self.current_spark()
        if spark is None:
            return ""
        rgb = self.base_rgb.copy()
        _draw_crosshair(rgb, spark.px, spark.py)
        return _data_url(rgb)

    def update_current(self, xs: float, ys: float) -> float:
        spark = self.current_spark()
        if spark is None:
            raise IndexError("No current spark")
        dist = compute_dist2memb(xs, ys, spark.px, spark.py, self.dx_um)
        spark.dist2memb = dist
        _set_field(self.sparks[spark.index], "dist2memb", dist)
        return dist

    def advance(self) -> None:
        self.current += 1
        if self.current >= len(self.accepted):
            self.save()
            self.finished.set()

    def save(self) -> None:
        if self.saved:
            return
        scipy.io.savemat(self.zdata_path, _clean_mat_dict(self.zdata))
        self.saved = True


HTML = r"""<!doctype html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modify dist2memb</title>
  <style>
REPORT_CSS

#main {
  height: calc(100vh - 43px);
  display: grid;
  grid-template-columns: minmax(240px, 1fr) 280px;
}
#viewer {
  overflow: auto;
  padding: 12px;
  background: var(--bg);
}
#canvas-wrap {
  position: relative;
  display: inline-block;
  border: 1px solid var(--border);
  background: #000;
}
#fluor-canvas {
  display: block;
  image-rendering: pixelated;
  cursor: crosshair;
}
#side {
  border-left: 1px solid var(--border);
  background: var(--surface);
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.metric {
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px;
}
.metric-label {
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .4px;
}
.metric-value {
  color: var(--accent);
  font-size: 16px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  margin-top: 3px;
}
.btn {
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--border);
  color: var(--text);
  cursor: pointer;
  font-size: 12px;
  padding: 8px 10px;
  text-align: center;
}
.btn.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #111;
  font-weight: 700;
}
#hint {
  color: var(--muted);
  line-height: 1.45;
}
  </style>
</head>
<body>
  <header id="hdr">
    <h1>Modify dist2memb</h1>
    <div class="kpi-bar">
      <div class="kpi"><span class="kpi-lbl">Progress</span><span class="kpi-val" id="progress">Spark - / -</span></div>
      <div class="kpi"><span class="kpi-lbl">DX</span><span class="kpi-val" id="dx">- um/px</span></div>
    </div>
    <button id="theme-btn" onclick="toggleTheme()">☀ Light</button>
  </header>
  <main id="main">
    <section id="viewer">
      <div id="canvas-wrap">
        <canvas id="fluor-canvas"></canvas>
      </div>
    </section>
    <aside id="side">
      <div class="metric">
        <div class="metric-label">Spark ID</div>
        <div class="metric-value" id="spark-id">-</div>
      </div>
      <div class="metric">
        <div class="metric-label">Current d2m</div>
        <div class="metric-value" id="d2m">- um</div>
      </div>
      <button class="btn primary" id="skip-btn">Skip (keep current)</button>
      <div id="hint">Click a membrane point in the image. The red line is the mask contour; the yellow crosshair marks the spark.</div>
    </aside>
  </main>
  <script>
function toggleTheme() {
  const root = document.documentElement;
  const theme = root.getAttribute('data-theme');
  root.setAttribute('data-theme', theme === 'light' ? 'dark' : 'light');
  document.getElementById('theme-btn').textContent =
    theme === 'light' ? '☀ Light' : '🌙 Dark';
}

let currentSpark = null;
const canvas = document.getElementById('fluor-canvas');
const ctx = canvas.getContext('2d');

function drawTextOverlay() {
  if (!currentSpark) return;
  const text = 'd2m = ' + currentSpark.dist2memb.toFixed(2) + ' um';
  ctx.font = '12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
  ctx.textBaseline = 'top';
  const w = ctx.measureText(text).width + 12;
  ctx.fillStyle = 'rgba(13, 17, 23, 0.72)';
  ctx.fillRect(6, 6, w, 24);
  ctx.strokeStyle = '#f0883e';
  ctx.strokeRect(6.5, 6.5, w - 1, 23);
  ctx.fillStyle = '#f0f6fc';
  ctx.fillText(text, 12, 12);
}

function showDone() {
  document.body.innerHTML = '<div style="padding:24px;font:14px sans-serif;color:var(--text);background:var(--bg);height:100vh">Finished. Saved updated MATLAB data.</div>';
}

async function loadSpark() {
  const res = await fetch('/spark');
  const data = await res.json();
  if (data.done) {
    showDone();
    return;
  }
  currentSpark = data.spark;
  canvas.width = data.width;
  canvas.height = data.height;
  document.getElementById('progress').textContent = 'Spark ' + data.position + ' / ' + data.total;
  document.getElementById('dx').textContent = data.dx_um.toFixed(4) + ' um/px';
  document.getElementById('spark-id').textContent = currentSpark.spark_id;
  document.getElementById('d2m').textContent = currentSpark.dist2memb.toFixed(2) + ' um';
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0);
    drawTextOverlay();
  };
  img.src = data.image;
}

canvas.addEventListener('click', async (event) => {
  if (!currentSpark) return;
  const rect = canvas.getBoundingClientRect();
  const xs = ((event.clientX - rect.left) * canvas.width / rect.width) + 1;
  const ys = ((event.clientY - rect.top) * canvas.height / rect.height) + 1;
  const res = await fetch('/update_d2m', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({spark_id: currentSpark.spark_id, xs, ys})
  });
  const data = await res.json();
  if (!data.ok) {
    alert(data.error || 'Could not update d2m');
    return;
  }
  if (data.done) {
    showDone();
    return;
  }
  await loadSpark();
});

document.getElementById('skip-btn').addEventListener('click', async () => {
  const res = await fetch('/skip', {method: 'POST'});
  const data = await res.json();
  if (data.done) {
    showDone();
    return;
  }
  await loadSpark();
});

loadSpark();
  </script>
</body>
</html>
"""


def _create_app(state: ModifyDist2MembraneState):
    try:
        from flask import Flask, jsonify, request
    except Exception as exc:  # pragma: no cover - depends on installed package
        raise RuntimeError("Flask is required for modify-d2m") from exc

    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML.replace("REPORT_CSS", REPORT_CSS)

    @app.get("/spark")
    def spark():
        current = state.current_spark()
        if current is None:
            return jsonify({"done": True})
        height, width = state.base_rgb.shape[:2]
        return jsonify(
            {
                "done": False,
                "position": state.current + 1,
                "total": len(state.accepted),
                "width": width,
                "height": height,
                "dx_um": state.dx_um,
                "image": state.image_for_current(),
                "spark": {
                    "number": current.number,
                    "spark_id": current.spark_id,
                    "px": current.px,
                    "py": current.py,
                    "dist2memb": current.dist2memb,
                },
            }
        )

    @app.post("/update_d2m")
    def update_d2m():
        payload = request.get_json(force=True, silent=True) or {}
        current = state.current_spark()
        if current is None:
            return jsonify({"ok": False, "error": "No current spark"}), 400
        if int(payload.get("spark_id", -1)) != current.spark_id:
            return jsonify({"ok": False, "error": "Stale spark id"}), 409
        try:
            xs = float(payload["xs"])
            ys = float(payload["ys"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "xs and ys must be numeric"}), 400
        dist = state.update_current(xs, ys)
        state.advance()
        return jsonify({"ok": True, "dist2memb": dist, "done": state.finished.is_set()})

    @app.post("/skip")
    def skip():
        state.advance()
        return jsonify({"ok": True, "done": state.finished.is_set()})

    return app


class _ServerThread(threading.Thread):
    def __init__(self, app, host: str, port: int) -> None:
        # Werkzeug belongs to the optional browser UI, not the numerical
        # compute_dist2memb helper.  Delaying the import keeps science-only
        # tests importable without web-server dependencies.
        from werkzeug.serving import make_server

        super().__init__(daemon=True)
        self.server = make_server(host, port, app)

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


def run_modify_dist2membrane(result_dir: str | os.PathLike[str] | None = None) -> Path:
    """Launch the local UI and block until the user finishes all accepted sparks."""
    selected_dir = Path(result_dir) if result_dir is not None else _find_result_dir()
    selected_dir = selected_dir.expanduser().resolve()
    state = ModifyDist2MembraneState(selected_dir)
    app = _create_app(state)
    server = _ServerThread(app, "127.0.0.1", PORT)
    server.start()
    url = f"http://localhost:{PORT}"
    print(f"Modify dist2memb UI: {url}", flush=True)
    webbrowser.open(url)

    try:
        while not state.finished.wait(timeout=0.25):
            time.sleep(0.01)
    finally:
        server.shutdown()
        server.join(timeout=5)
    return state.zdata_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manually edit spark dist2memb values.")
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help="Results folder. If omitted, a folder picker is shown.",
    )
    args = parser.parse_args(argv)
    saved = run_modify_dist2membrane(args.result_dir)
    print(f"Saved updated distances to {saved}")


if __name__ == "__main__":  # pragma: no cover
    main()
