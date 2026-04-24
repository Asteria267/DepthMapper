"""
╔══════════════════════════════════════════════════════════════════════╗
║   BUILDCORED ORCAS — Day 18: DepthMapper                            ║
║   Rose Quartz Edition  ·  MiDaS Monocular Depth Estimation          ║
║   Webcam → Depth Heatmap → Point Cloud CSV → Histogram              ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT IS MONOCULAR DEPTH ESTIMATION?
─────────────────────────────────────
A real depth sensor (LiDAR, ToF, structured light) shoots photons
at the scene and measures how long they take to bounce back.

MiDaS does something wilder: it looks at a SINGLE 2D RGB image
and uses a neural network trained on millions of real depth scans
to *predict* the depth of every pixel. No special hardware needed.

Output: a depth map — same H×W as your image, but each pixel
holds a float representing "relative distance from camera."

This is the software equivalent of:
  • Intel RealSense D435 (structured light)
  • SICK LiDAR (time of flight)
  • iPhone LiDAR Scanner (dToF)
  • Velodyne VLP-16 (spinning lidar)

HARDWARE BRIDGE:
  In v2.0 you'd replace get_depth_frame() with a RealSense SDK call.
  The rest of the pipeline — colorize, histogram, CSV export — stays
  IDENTICAL. That's the power of building the abstraction right.

BACKENDS (auto-detected, best first):
  1. PyTorch + MiDaS DPT-Small   — best quality, ~2-5 FPS on CPU
  2. PyTorch + MiDaS MiDaS-Small — slightly lighter
  3. ONNX Runtime + MiDaS-Small  — no torch needed, similar speed
  4. Gaussian blur fallback       — no ML at all, for demos

Run:
    pip install opencv-python numpy matplotlib torch torchvision
    # OR for ONNX fallback:
    pip install opencv-python numpy matplotlib onnxruntime

    python depthmapper.py

Controls:
    SPACE   — export current frame as point_cloud.csv
    H       — toggle histogram window
    Q / ESC — quit

Day 18 — BUILDCORED ORCAS
"""

# ══════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════

import sys
import os
import time
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation

# ══════════════════════════════════════════════════════════════════════
# ROSE QUARTZ PALETTE  (consistent with Day 17 v3)
# ══════════════════════════════════════════════════════════════════════

RQ = {
    "bg":        "#0d0911",
    "panel":     "#17111d",
    "panel2":    "#1f1628",
    "grid":      "#2b2033",
    "text":      "#f7d9ea",
    "dim":       "#9b7f92",
    "dimmer":    "#4a3545",
    "pink":      "#ff4fa3",
    "pink_glow": "#ff8fc4",
    "pink_dim":  "#3d0f24",
    "violet":    "#b060ff",
    "viol_glow": "#d0a0ff",
    "soft":      "#ffc1dd",
    "gold":      "#ffd700",
    "green":     "#a8ff78",
    "cyan":      "#78ffd6",
}

# OpenCV depth colormap — MAGMA looks stunning & is perceptually uniform
DEPTH_COLORMAP = cv2.COLORMAP_MAGMA

# ══════════════════════════════════════════════════════════════════════
# DEVICE AUTO-DETECT
# ══════════════════════════════════════════════════════════════════════

def detect_device():
    """Pick best available compute device. Returns (device_str, label)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", f"CUDA ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "Apple MPS"
        return "cpu", "CPU"
    except ImportError:
        return "cpu", "CPU (torch not found)"

# ══════════════════════════════════════════════════════════════════════
# DEPTH BACKEND — tries best option, falls back gracefully
# ══════════════════════════════════════════════════════════════════════

class DepthBackend:
    """
    Abstract depth estimator.
    Subclasses implement estimate(bgr_frame) → float32 depth map [0,1].
    """
    name = "base"

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MiDaSBackend(DepthBackend):
    """
    PyTorch MiDaS — best quality.
    Downloads ~80-300 MB on first run (MiDaS-small).
    """
    name = "MiDaS (PyTorch)"

    def __init__(self, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)

        print("  [MiDaS] Loading model via torch.hub … (first run downloads ~80MB)")
        # DPT_Small is faster than DPT_Large, good quality
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                trust_repo=True, verbose=False
            )
            transforms_hub = torch.hub.load(
                "intel-isl/MiDaS", "transforms",
                trust_repo=True, verbose=False
            )
            self.transform = transforms_hub.small_transform
        except Exception as e:
            raise RuntimeError(f"torch.hub failed: {e}")

        self.model.to(self.device)
        self.model.eval()
        print(f"  [MiDaS] Model ready on {device}")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        import torch
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=bgr_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth.astype(np.float32)


class ONNXBackend(DepthBackend):
    """
    ONNX Runtime fallback — no PyTorch needed.
    Downloads MiDaS-small ONNX model on first run (~25 MB).
    """
    name = "MiDaS ONNX (onnxruntime)"

    MODEL_URL  = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
    MODEL_FILE = "midas_small.onnx"
    INPUT_SIZE = (256, 256)   # MiDaS-small input

    def __init__(self):
        import onnxruntime as ort
        if not os.path.exists(self.MODEL_FILE):
            self._download()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.MODEL_FILE, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"  [ONNX] Session ready — provider: {self.session.get_providers()[0]}")

    def _download(self):
        import urllib.request
        print(f"  [ONNX] Downloading MiDaS-small (~25 MB) …")
        urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_FILE)
        print("  [ONNX] Download complete.")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        h, w = bgr_frame.shape[:2]
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp  = cv2.resize(rgb, self.INPUT_SIZE).astype(np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp  = (inp - mean) / std
        inp  = inp.transpose(2, 0, 1)[np.newaxis]   # NCHW

        raw = self.session.run(None, {self.input_name: inp})[0].squeeze()
        depth = cv2.resize(raw, (w, h))

        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth.astype(np.float32)


class GaussianFallback(DepthBackend):
    """
    Zero-dependency artistic fallback.
    Uses edge detection + blur to fake plausible depth-like gradients.
    NOT real depth estimation — clearly labelled in the UI.
    Use only when no ML runtime is available.
    """
    name = "Gaussian Fallback (no ML)"

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Laplacian edges → regions of change → proxy for depth variation
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
        lap = np.abs(lap)
        blurred = cv2.GaussianBlur(lap, (61, 61), 0)
        # Add radial vignette (centre closer, edges farther — common heuristic)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        radial = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        combined = 0.6 * blurred + 0.4 * (1.0 - radial) * gray.max()
        d_min, d_max = combined.min(), combined.max()
        if d_max - d_min > 1e-6:
            combined = (combined - d_min) / (d_max - d_min)
        return combined.astype(np.float32)


def build_backend() -> DepthBackend:
    """
    Try backends best-to-worst. Return the first that works.
    Prints clear status so the user knows what's running.
    """
    device_str, device_label = detect_device()
    print(f"\n  Device detected: {device_label}")

    # ── 1. Try PyTorch MiDaS ──────────────────────────────────────────
    try:
        import torch   # noqa
        print("  Trying PyTorch MiDaS …")
        backend = MiDaSBackend(device=device_str)
        return backend
    except ImportError:
        print("  PyTorch not installed — skipping.")
    except Exception as e:
        print(f"  MiDaS torch failed: {e}")

    # ── 2. Try ONNX Runtime ───────────────────────────────────────────
    try:
        import onnxruntime   # noqa
        print("  Trying ONNX Runtime …")
        backend = ONNXBackend()
        return backend
    except ImportError:
        print("  onnxruntime not installed — skipping.")
    except Exception as e:
        print(f"  ONNX failed: {e}")

    # ── 3. Gaussian fallback ──────────────────────────────────────────
    print("  ⚠  No ML runtime found. Using Gaussian fallback.")
    print("     Install torch or onnxruntime for real depth estimation.")
    return GaussianFallback()


# ══════════════════════════════════════════════════════════════════════
# POINT CLOUD EXPORTER
# ══════════════════════════════════════════════════════════════════════

def export_point_cloud(depth_map: np.ndarray, filename: str = "point_cloud.csv"):
    """..."""
    h, w = depth_map.shape
    step = 4

    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ds = depth_map[::step, ::step]
    depth_mm = (ds * 8000).astype(np.int32)
    data = np.column_stack([xs.ravel(), ys.ravel(), ds.ravel(), depth_mm.ravel()])

    header = "x,y,depth_norm,depth_mm"
    np.savetxt(filename, data, delimiter=",", header=header,
               comments="", fmt=["%d", "%d", "%.4f", "%d"])

    print(f"  ✅ Point cloud saved → {filename}  ({len(xs.ravel()):,} points)")
    return len(xs.ravel())


# ══════════════════════════════════════════════════════════════════════
# MATPLOTLIB DASHBOARD
# ══════════════════════════════════════════════════════════════════════

class DepthDashboard:
    """
    Real-time matplotlib dashboard:
      Left  — original webcam feed (BGR→RGB)
      Right — depth heatmap (MAGMA colourmap)
      Bottom — live depth histogram + stats
    """

    def __init__(self, backend: DepthBackend):
        self.backend = backend

        # ── Figure ────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(16, 9), facecolor=RQ["bg"])
        self.fig.canvas.manager.set_window_title(
            "DepthMapper — Day 18 | BUILDCORED ORCAS | Rose Quartz"
        )

        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            height_ratios=[3.8, 2],
            width_ratios=[3, 3, 2.2],
            hspace=0.42, wspace=0.28,
            left=0.04, right=0.97,
            top=0.91, bottom=0.05,
        )

        # ── Title ─────────────────────────────────────────────────────
        self.fig.text(0.5, 0.965, "DepthMapper",
                      ha="center", color=RQ["pink"],
                      fontsize=22, fontweight="bold", family="monospace")
        self.fig.text(0.5, 0.943,
                      f"Monocular Depth Estimation  ·  {backend.name}  ·  "
                      "BUILDCORED ORCAS — Day 18",
                      ha="center", color=RQ["dim"], fontsize=8.5)
        self.fig.add_artist(plt.Line2D(
            [0.04, 0.97], [0.934, 0.934],
            transform=self.fig.transFigure,
            color=RQ["grid"], linewidth=1
        ))

        # ── Webcam panel ──────────────────────────────────────────────
        self.ax_cam = self.fig.add_subplot(gs[0, 0])
        self._style_ax(self.ax_cam, "Webcam Feed", RQ["pink"])
        self.im_cam = self.ax_cam.imshow(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.ax_cam.axis("off")

        # ── Depth heatmap panel ───────────────────────────────────────
        self.ax_depth = self.fig.add_subplot(gs[0, 1])
        self._style_ax(self.ax_depth, "Depth Heatmap  (MAGMA)", RQ["violet"])
        self.im_depth = self.ax_depth.imshow(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.ax_depth.axis("off")

        # Colorbar
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        cax = self.fig.add_axes([0.655, 0.52, 0.012, 0.35])
        cax.set_facecolor(RQ["bg"])
        cb = ColorbarBase(cax, cmap=cm.magma,
                          norm=Normalize(vmin=0, vmax=1),
                          orientation="vertical")
        cb.ax.tick_params(labelsize=7, colors=RQ["dim"])
        cb.set_label("Near → Far", fontsize=7, color=RQ["dim"])

        # ── Stats card (top-right) ────────────────────────────────────
        self.ax_stats = self.fig.add_subplot(gs[0, 2])
        self.ax_stats.set_facecolor(RQ["panel"])
        self.ax_stats.axis("off")
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self._style_ax(self.ax_stats, "Live Stats", RQ["gold"])

        stat_defs = [
            ("fps",    0.82, "FPS",          RQ["green"]),
            ("mean",   0.62, "Mean Depth",   RQ["pink"]),
            ("near",   0.42, "Nearest %",    RQ["viol_glow"]),
            ("far",    0.22, "Farthest %",   RQ["soft"]),
            ("frames", 0.04, "Frames",       RQ["dim"]),
        ]
        self.stat_texts = {}
        for key, y, label, color in stat_defs:
            card = FancyBboxPatch(
                (0.06, y - 0.03), 0.88, 0.165,
                boxstyle="round,pad=0.02",
                linewidth=1, edgecolor=color,
                facecolor=RQ["panel2"], alpha=0.9,
                transform=self.ax_stats.transAxes, zorder=2
            )
            self.ax_stats.add_patch(card)
            self.ax_stats.text(0.12, y + 0.10, label,
                               transform=self.ax_stats.transAxes,
                               color=RQ["dim"], fontsize=7.5, zorder=3)
            t = self.ax_stats.text(
                0.5, y + 0.04, "—",
                transform=self.ax_stats.transAxes,
                color=color, fontsize=19,
                fontweight="bold", fontfamily="monospace",
                ha="center", zorder=3
            )
            self.stat_texts[key] = t

        # ── Histogram (bottom, span cols 0-1) ────────────────────────
        self.ax_hist = self.fig.add_subplot(gs[1, 0:2])
        self._style_ax(self.ax_hist,
                       "Depth Distribution Histogram  (live)", RQ["pink"])
        self.ax_hist.set_facecolor(RQ["panel"])
        self.ax_hist.set_xlabel("Normalised Depth  (0 = near, 1 = far)",
                                color=RQ["dim"], fontsize=8)
        self.ax_hist.set_ylabel("Pixel Count", color=RQ["dim"], fontsize=8)
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.tick_params(colors=RQ["dimmer"], labelsize=7)
        for s in self.ax_hist.spines.values():
            s.set_color(RQ["grid"])
        self.ax_hist.grid(True, alpha=0.2, color=RQ["grid"])

        # Pre-build histogram bar containers
        self._n_bins = 60
        self._bin_edges = np.linspace(0, 1, self._n_bins + 1)
        self._bar_x = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        bar_width = (self._bin_edges[1] - self._bin_edges[0]) * 0.9

        # Use magma colour for each bar (matches depth heatmap visually)
        import matplotlib.cm as cm
        bar_colors = [cm.magma(v) for v in self._bar_x]
        self._bars = self.ax_hist.bar(
            self._bar_x, np.zeros(self._n_bins),
            width=bar_width, color=bar_colors, alpha=0.85
        )
        self.ax_hist.set_ylim(0, 1)  # will be rescaled each frame

        # Mean line on histogram
        self._hist_mean_line = self.ax_hist.axvline(
            0.5, color=RQ["pink"], lw=1.6, ls="--", alpha=0.8
        )
        self._hist_mean_txt = self.ax_hist.text(
            0.52, 0.92, "", transform=self.ax_hist.transAxes,
            color=RQ["pink"], fontsize=8
        )

        # ── Code export panel (bottom-right) ─────────────────────────
        self.ax_code = self.fig.add_subplot(gs[1, 2])
        self.ax_code.set_facecolor(RQ["panel"])
        self.ax_code.axis("off")
        self._style_ax(self.ax_code, "Hardware Bridge", RQ["cyan"])
        self._code_txt = self.ax_code.text(
            0.05, 0.92, self._hardware_bridge_text(),
            transform=self.ax_code.transAxes,
            ha="left", va="top",
            fontsize=6.8, color=RQ["soft"], family="monospace",
            linespacing=1.5
        )

        # ── Controls hint ─────────────────────────────────────────────
        self.fig.text(0.5, 0.005,
                      "SPACE = export CSV point cloud   |   "
                      "H = toggle histogram   |   Q / ESC = quit",
                      ha="center", fontsize=7, color=RQ["dimmer"])

        # ── Export notification text ──────────────────────────────────
        self._export_note = self.fig.text(
            0.5, 0.021, "",
            ha="center", fontsize=8,
            color=RQ["green"], fontweight="bold"
        )
        self._export_note_timer = 0.0

        # ── State ─────────────────────────────────────────────────────
        self._frame_count   = 0
        self._last_time     = time.time()
        self._fps           = 0.0
        self._last_depth    = None
        self._show_hist     = True
        self._running       = True

    # ── helpers ───────────────────────────────────────────────────────

    def _style_ax(self, ax, title, color):
        ax.set_facecolor(RQ["panel"])
        ax.set_title(title, color=color, fontsize=10,
                     fontweight="bold", pad=5, loc="left")

    def _hardware_bridge_text(self):
        return (
            "— v2.0 Hardware Swap —\n\n"
            "# RealSense D435\n"
            "import pyrealsense2 as rs\n"
            "pipe = rs.pipeline()\n"
            "pipe.start()\n"
            "frames = pipe.wait_for_frames()\n"
            "depth = frames.get_depth_frame()\n"
            "d = np.asarray(depth.get_data())\n\n"
            "# LiDAR (ROS)\n"
            "from sensor_msgs.msg import LaserScan\n"
            "# callback(msg): msg.ranges\n\n"
            "# iPhone LiDAR (ARKit)\n"
            "# → export .PLY from SceneKit\n"
            "# → load with open3d"
        )

    # ── public ────────────────────────────────────────────────────────

    def update(self, bgr_frame: np.ndarray, depth_map: np.ndarray):
        """Refresh all panels with new frame data."""
        self._frame_count += 1

        # FPS
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self._fps = 0.85 * self._fps + 0.15 * (1.0 / (dt + 1e-9))

        self._last_depth = depth_map.copy()

        # ── Webcam panel ──────────────────────────────────────────────
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.im_cam.set_data(rgb)

        # ── Depth heatmap ─────────────────────────────────────────────
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        coloured_bgr = cv2.applyColorMap(depth_uint8, DEPTH_COLORMAP)
        coloured_rgb = cv2.cvtColor(coloured_bgr, cv2.COLOR_BGR2RGB)
        self.im_depth.set_data(coloured_rgb)

        # ── Histogram ─────────────────────────────────────────────────
        if self._show_hist:
            counts, _ = np.histogram(depth_map.ravel(), bins=self._bin_edges)
            max_count = counts.max() if counts.max() > 0 else 1
            for bar, h in zip(self._bars, counts):
                bar.set_height(h)
            self.ax_hist.set_ylim(0, max_count * 1.12)

            mean_d = depth_map.mean()
            self._hist_mean_line.set_xdata([mean_d, mean_d])
            self._hist_mean_txt.set_text(f"μ = {mean_d:.3f}")

        # ── Stats cards ───────────────────────────────────────────────
        mean_d  = depth_map.mean()
        near_pct = (depth_map < 0.25).mean() * 100
        far_pct  = (depth_map > 0.75).mean() * 100

        self.stat_texts["fps"].set_text(f"{self._fps:.1f}")
        self.stat_texts["mean"].set_text(f"{mean_d:.3f}")
        self.stat_texts["near"].set_text(f"{near_pct:.1f}%")
        self.stat_texts["far"].set_text(f"{far_pct:.1f}%")
        self.stat_texts["frames"].set_text(str(self._frame_count))

        # ── Export notification fade ──────────────────────────────────
        if self._export_note_timer > 0:
            self._export_note_timer -= dt
            if self._export_note_timer <= 0:
                self._export_note.set_text("")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def notify_export(self, n_points: int, filename: str):
        msg = f"✅  Exported {n_points:,} points → {filename}"
        self._export_note.set_text(msg)
        self._export_note_timer = 4.0

    def toggle_hist(self):
        self._show_hist = not self._show_hist
        self.ax_hist.set_visible(self._show_hist)

    @property
    def last_depth(self):
        return self._last_depth


# ══════════════════════════════════════════════════════════════════════
# WEBCAM CAPTURE THREAD
# ══════════════════════════════════════════════════════════════════════

class WebcamCapture:
    """
    Captures frames in a background thread so the main loop
    doesn't block waiting for the camera.
    """
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Don't buffer old frames

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open webcam. Check it's connected and not in use."
            )

        self._lock  = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"  Webcam opened (index {camera_index}) — "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   DepthMapper — Day 18  ·  BUILDCORED ORCAS                 ║")
    print("║   Rose Quartz Edition  ·  Monocular Depth Estimation         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  SPACE  → export frame as point_cloud.csv                   ║")
    print("║  H      → toggle histogram                                  ║")
    print("║  Q/ESC  → quit                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Backend ───────────────────────────────────────────────────────
    print("\n▶ Initialising depth backend …")
    backend = build_backend()
    print(f"  Active backend: {backend.name}\n")

    # ── Webcam ────────────────────────────────────────────────────────
    print("▶ Opening webcam …")
    try:
        cam = WebcamCapture(camera_index=0)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Wait for first frame
    print("  Waiting for first frame …")
    for _ in range(50):
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("  ❌ No frames from webcam.")
        cam.release()
        sys.exit(1)

    print("  ✅ Webcam ready.\n")

    # ── Dashboard ─────────────────────────────────────────────────────
    print("▶ Building dashboard …")
    plt.ion()
    dashboard = DepthDashboard(backend)
    print("  ✅ Dashboard ready.\n")
    print("▶ Running … (close window or press Q to stop)")

    # ── Keyboard handler (matplotlib) ─────────────────────────────────
    def on_key(event):
        if event.key in ("q", "escape"):
            dashboard._running = False
            plt.close("all")

        elif event.key == " ":
            d = dashboard.last_depth
            if d is not None:
                n = export_point_cloud(d, "point_cloud.csv")
                dashboard.notify_export(n, "point_cloud.csv")
            else:
                print("  No depth frame yet.")

        elif event.key in ("h", "H"):
            dashboard.toggle_hist()

    dashboard.fig.canvas.mpl_connect("key_press_event", on_key)

    # ── Main loop ─────────────────────────────────────────────────────
    frame_interval = 1.0 / 30.0   # target 30 FPS display
    last_depth_time = 0.0
    depth_interval  = 0.25         # run inference ~4× per second (adjustable)

    depth_map = np.full((480, 640), 0.5, dtype=np.float32)   # placeholder

    try:
        while dashboard._running:
            loop_start = time.time()

            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Run depth inference on schedule
            now = time.time()
            if now - last_depth_time >= depth_interval:
                depth_map = backend.estimate(frame)
                last_depth_time = now

            # Update display
            dashboard.update(frame, depth_map)

            # Pace the loop
            elapsed = time.time() - loop_start
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        print("\n▶ Shutting down …")
        cam.release()
        plt.close("all")
        print("  ✅ Done. See you on Day 19! 🐋\n")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
