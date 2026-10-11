"""
Genetic Algorithm Image Evolution Studio  (v14.1 — Complete)
All features: 18 shapes, HSV, batch processing, timing, temporal coherence,
algorithm improvements, performance optimizations, professional MP4 output.
"""

import sys, os, json, time, random, threading, tempfile, requests, csv, re, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
import colorsys
import wave
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTextEdit, QLineEdit, QFileDialog,
    QGroupBox, QFormLayout, QSplitter, QMessageBox, QCheckBox,
    QComboBox, QTabWidget, QStatusBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QInputDialog
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QSize, QTimer, QSettings
)
from PySide6.QtGui import (
    QPixmap, QImage, QAction, QDragEnterEvent, QDropEvent,
    QPainter, QPen, QColor, QKeySequence, QFont
)

import cv2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class ImageEntry:
    def __init__(self, url: str, caption: str = ""):
        self.url = url.strip()
        self.caption = caption.strip()
        self.target_array = None
        self.loaded = False
        self.preview_pixmap = None
        self.thumbnail = None
        self.status = "pending"
        self.current_gen = 0
        self.total_gens = 0
        self.current_mse = 0.0
        self.final_mse = 0.0
        self.gens_run = 0
        self.output_path = None
        self.png_path = None
        self.started_at = None
        self.finished_at = None


# ---------------------------------------------------------------------------
# Live fitness graph
# ---------------------------------------------------------------------------
class FitnessGraphWidget(QWidget):
    def __init__(self, parent=None, max_points=800):
        super().__init__(parent)
        self.max_points = max_points
        self.data = []
        self.setMinimumHeight(180)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        self.title = "Fitness (MSE)"

    def add_point(self, gen, mse):
        self.data.append((gen, max(1e-8, mse)))
        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points:]
        self.update()

    def clear(self):
        self.data = []
        self.update()

    def paintEvent(self, event):
        if not self.data:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        margin = 40
        painter.fillRect(0, 0, w, h, QColor(26, 26, 26))
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(5):
            y = margin + (h - 2 * margin) * i / 4
            painter.drawLine(margin, int(y), w - 10, int(y))
        gens = [d[0] for d in self.data]
        mses = [d[1] for d in self.data]
        min_gen, max_gen = min(gens), max(gens)
        log_mses = [math.log10(m) for m in mses]
        min_log, max_log = min(log_mses), max(log_mses)
        if max_log - min_log < 0.1:
            max_log = min_log + 0.1
        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Consolas", 8))
        for i in range(5):
            log_val = max_log - (max_log - min_log) * i / 4
            val = 10 ** log_val
            y = margin + (h - 2 * margin) * i / 4
            painter.drawText(2, int(y) + 4, f"{val:.4f}")
        plot_w = w - margin - 10
        plot_h = h - 2 * margin
        path_points = []
        for (g, m) in self.data:
            x = margin + plot_w * (g - min_gen) / max(1, max_gen - min_gen)
            log_m = math.log10(max(1e-8, m))
            y = margin + plot_h * (1 - (log_m - min_log) / (max_log - min_log))
            path_points.append((int(x), int(y)))
        if len(path_points) > 1:
            painter.setPen(QPen(QColor(100, 200, 255), 2))
            for i in range(len(path_points) - 1):
                painter.drawLine(path_points[i][0], path_points[i][1],
                                 path_points[i+1][0], path_points[i+1][1])
            last = path_points[-1]
            painter.setBrush(QColor(255, 100, 100))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(last[0] - 3, last[1] - 3, 6, 6)
        painter.setPen(QColor(220, 220, 220))
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.drawText(margin, 15, self.title)
        if self.data:
            painter.drawText(w - 150, 15, f"Current: {self.data[-1][1]:.6f}")
        painter.end()


# ---------------------------------------------------------------------------
# Timing controller
# ---------------------------------------------------------------------------
class TimingController:
    STOP_MODES = ["max_gens", "target_mse", "convergence", "combined"]
    TRANSITION_TYPES = ["cut", "crossfade", "fade_black", "wipe"]
    SPEED_CURVES = ["linear", "ease_in", "ease_out", "ease_in_out"]

    def __init__(self, params):
        self.params = params
        self.stop_mode = params.get("stop_mode", "target_mse")
        self.max_gens = int(params.get("generations", 1000))
        self.target_mse = float(params.get("target_mse", 0.01))
        self.convergence_threshold = float(params.get("convergence_threshold", 1e-5))
        self.convergence_patience = int(params.get("convergence_patience", 30))
        self.segment_duration = float(params.get("segment_duration", 6.0))
        self.fps = int(params.get("fps", 20))
        self.hold_seconds = float(params.get("hold_seconds", 0.5))
        self.transition_type = params.get("transition_type", "cut")
        self.transition_seconds = float(params.get("transition_seconds", 0.5))
        self.speed_curve = params.get("speed_curve", "linear")
        self.intro_enabled = params.get("intro_enabled", False)
        self.intro_text = params.get("intro_text", "")
        self.intro_seconds = float(params.get("intro_seconds", 2.0))
        self.outro_enabled = params.get("outro_enabled", False)
        self.outro_text = params.get("outro_text", "")
        self.outro_seconds = float(params.get("outro_seconds", 2.0))

    def should_stop_ga(self, gen, current_mse, fitness_history):
        if self.stop_mode == "max_gens":
            return gen >= self.max_gens
        if self.stop_mode == "target_mse":
            return current_mse <= self.target_mse
        if self.stop_mode == "convergence":
            return self._check_convergence(fitness_history)
        if self.stop_mode == "combined":
            if current_mse <= self.target_mse: return True
            if gen >= self.max_gens: return True
            if self._check_convergence(fitness_history): return True
            return False
        return False

    def _check_convergence(self, fitness_history):
        if len(fitness_history) < self.convergence_patience:
            return False
        recent = fitness_history[-self.convergence_patience:]
        return (recent[0] - recent[-1]) < self.convergence_threshold

    def get_target_frames_for_segment(self):
        return max(1, int(self.segment_duration * self.fps))

    def resample_frames_to_duration(self, frames):
        target_frames = self.get_target_frames_for_segment()
        if len(frames) == 0: return frames
        if len(frames) == 1: return [frames[0]] * target_frames
        resampled = []
        for i in range(target_frames):
            t_norm = i / max(1, target_frames - 1)
            t_curved = self.apply_speed_curve(t_norm)
            src_idx = t_curved * (len(frames) - 1)
            idx_low = int(np.floor(src_idx))
            idx_high = min(len(frames) - 1, int(np.ceil(src_idx)))
            frac = src_idx - idx_low
            if idx_low == idx_high or frac < 0.01:
                resampled.append(frames[idx_low])
            else:
                blended = (frames[idx_low].astype(np.float32) * (1 - frac) +
                           frames[idx_high].astype(np.float32) * frac).astype(np.uint8)
                resampled.append(blended)
        return resampled

    def apply_speed_curve(self, progress):
        if self.speed_curve == "ease_in": return progress ** 2
        if self.speed_curve == "ease_out": return 1 - (1 - progress) ** 2
        if self.speed_curve == "ease_in_out": return 3 * progress ** 2 - 2 * progress ** 3
        return progress

    def get_hold_frames(self):
        return max(0, int(self.hold_seconds * self.fps))

    def get_transition_frames(self):
        return max(0, int(self.transition_seconds * self.fps))


# ---------------------------------------------------------------------------
# Transitions and cards
# ---------------------------------------------------------------------------
class TransitionRenderer:
    @staticmethod
    def crossfade(a, b, t):
        return (a * (1 - t) + b * t).astype(np.uint8)

    @staticmethod
    def fade_black(frame, t):
        factor = 1 - (t * 2) if t < 0.5 else (t - 0.5) * 2
        return (frame * factor).astype(np.uint8)

    @staticmethod
    def wipe(a, b, t, direction="left"):
        h, w = a.shape[:2]
        result = a.copy()
        if direction == "left":
            split = int(w * t); result[:, :split] = b[:, :split]
        elif direction == "right":
            split = int(w * (1 - t)); result[:, split:] = b[:, split:]
        elif direction == "down":
            split = int(h * t); result[:split, :] = b[:split, :]
        return result


class CardGenerator:
    @staticmethod
    def generate(text, width, height, duration_sec, fps, bg_color=(20, 20, 30),
                 text_color=(255, 255, 255)):
        frames = []
        n_frames = max(1, int(duration_sec * fps))
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except Exception:
            try: font = ImageFont.truetype("arial.ttf", 32)
            except Exception: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - tw) // 2, (height - th) // 2), text, fill=text_color, font=font)
        base_arr = np.asarray(img)
        fade_frames = min(n_frames // 4, fps)
        for i in range(n_frames):
            if i < fade_frames: factor = i / fade_frames
            elif i > n_frames - fade_frames: factor = (n_frames - i) / fade_frames
            else: factor = 1.0
            frames.append((base_arr * factor).astype(np.uint8))
        return frames


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
class JsonImageLoader:
    URL_KEYS = {"url", "src", "image", "link", "href"}
    CAPTION_KEYS = {"caption", "title", "description", "desc", "text", "label"}

    @classmethod
    def load_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.parse(data)

    @classmethod
    def parse(cls, data):
        entries = []
        if isinstance(data, dict) and "images" in data:
            return cls.parse(data["images"])
        if isinstance(data, dict):
            url = cls._find_key(data, cls.URL_KEYS)
            if url:
                entries.append(ImageEntry(url, cls._find_key(data, cls.CAPTION_KEYS) or ""))
                return entries
            raise ValueError("JSON object must contain a URL key")
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    entries.append(ImageEntry(item, ""))
                elif isinstance(item, dict):
                    url = cls._find_key(item, cls.URL_KEYS)
                    if not url: raise ValueError(f"Item {idx} missing URL")
                    entries.append(ImageEntry(url, cls._find_key(item, cls.CAPTION_KEYS) or ""))
            return entries
        raise ValueError(f"Unsupported JSON root: {type(data).__name__}")

    @staticmethod
    def _find_key(d, candidates):
        lower_map = {k.lower(): v for k, v in d.items()}
        for c in candidates:
            if c in lower_map and isinstance(lower_map[c], str) and lower_map[c].strip():
                return lower_map[c].strip()
        return None


class CsvImageLoader:
    @staticmethod
    def load_file(path):
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = None
                for k in ("url", "src", "image", "link", "href"):
                    if k in row and row[k].strip():
                        url = row[k].strip(); break
                if not url: continue
                caption = ""
                for k in ("caption", "title", "description", "desc", "text", "label", "name"):
                    if k in row and row[k].strip():
                        caption = row[k].strip(); break
                entries.append(ImageEntry(url, caption))
        return entries


class UrlPatternExpander:
    RANGE_RE = re.compile(r"\{(\d+)\.\.(\d+)\}")
    PADDED_RANGE_RE = re.compile(r"\{(0\d+)\.\.(0\d+)\}")
    LIST_RE = re.compile(r"\{([^{}]+,[^{}]+)\}")

    @classmethod
    def expand(cls, pattern):
        results = [pattern]
        m = cls.PADDED_RANGE_RE.search(pattern)
        if m:
            start_s, end_s = m.group(1), m.group(2)
            width = len(start_s)
            start, end = int(start_s), int(end_s)
            step = 1 if start <= end else -1
            return [r[:m.start()] + str(i).zfill(width) + r[m.end():]
                    for r in results for i in range(start, end + step, step)]
        m = cls.RANGE_RE.search(pattern)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            step = 1 if start <= end else -1
            return [r[:m.start()] + str(i) + r[m.end():]
                    for r in results for i in range(start, end + step, step)]
        m = cls.LIST_RE.search(pattern)
        if m:
            items = [x.strip() for x in m.group(1).split(",")]
            return [r[:m.start()] + item + r[m.end():] for r in results for item in items]
        return results


class OutputNamer:
    @staticmethod
    def generate(template, entry, idx, params, preset_name="custom"):
        name = entry.caption or Path(entry.url).stem or f"image_{idx}"
        name = re.sub(r'[\\/*?:"<>|]', "_", name)[:50]
        now = datetime.now()
        result = template
        result = result.replace("{name}", name)
        result = result.replace("{idx}", f"{idx:03d}")
        result = result.replace("{shape}", params.get("shape_mode", "shape"))
        result = result.replace("{preset}", preset_name)
        result = result.replace("{date}", now.strftime("%Y%m%d"))
        result = result.replace("{time}", now.strftime("%H%M%S"))
        result = result.replace("{colorspace}", params.get("color_space", "rgb"))
        return result


class ImageDownloader:
    @staticmethod
    def download(url, timeout=15, max_retries=3):
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        raise last_error


# ---------------------------------------------------------------------------
# Presets (v14.1 — web/hd only)
# ---------------------------------------------------------------------------
BUILTIN_PRESETS = {
    "Fast Preview": {
        "canvas_w": 128, "canvas_h": 128, "num_shapes": 50,
        "population_size": 40, "mutation_rate": 0.03, "elite_count": 2,
        "alpha": 140, "generations": 500, "segment_duration": 3.0, "fps": 15,
        "display_w": 320, "display_h": 240, "n_sides": 5,
        "shape_mode": "triangle", "use_smart_init": True,
        "use_hill_climb": True, "hill_climb_steps": 10,
        "use_adaptive_mutation": True, "use_ssim": False,
        "use_multiscale": False, "multiscale_ratio": 0.3, "color_space": "rgb",
        "stop_mode": "target_mse", "target_mse": 0.02,
        "convergence_threshold": 1e-5, "convergence_patience": 20,
        "hold_seconds": 0.5, "transition_type": "cut", "transition_seconds": 0.0,
        "speed_curve": "linear", "intro_enabled": False, "outro_enabled": False,
        "seed": 0, "export_png": False,
        "fitness_mode": "animated",
        "camera_mode": "subtle",
        "temporal_smooth": False, "smooth_strength": 0.3,
        "temporal_coherence": 0.1, "inheritance_strength": 0.5,
        "progressive_shapes": False, "motion_blur": False,
        "motion_blur_strength": 0.3, "keyframe_interval": 3,
        "use_residual_guidance": True, "use_multiscale_fitness": False,
        "use_annealing": True, "use_de_blend": True,
        "use_shape_split": False, "allow_dynamic_shape_count": False,
        "use_fitness_cache": True, "use_diversity_injection": True,
        "use_importance_tracking": True, "use_gradient_mutation": True,
        "evaluation_scale": 0.25, "enable_roi": False, "roi_weight": 1.5,
        "quality_tier": "web", "generate_per_image_mp4": False,
        "generate_thumbnail": False, "post_sharpen": 0.0,
        "post_color_grade": "none", "post_vignette": 0.0,
        "post_film_grain": 0.0, "caption_style": "none", "audio_mood": "ambient",
    },
    "Balanced": {
        "canvas_w": 192, "canvas_h": 192, "num_shapes": 80,
        "population_size": 80, "mutation_rate": 0.02, "elite_count": 4,
        "alpha": 128, "generations": 1000, "segment_duration": 6.0, "fps": 20,
        "display_w": 480, "display_h": 360, "n_sides": 5,
        "shape_mode": "triangle", "use_smart_init": True,
        "use_hill_climb": True, "hill_climb_steps": 20,
        "use_adaptive_mutation": True, "use_ssim": False,
        "use_multiscale": False, "multiscale_ratio": 0.3, "color_space": "rgb",
        "stop_mode": "target_mse", "target_mse": 0.01,
        "convergence_threshold": 1e-5, "convergence_patience": 30,
        "hold_seconds": 1.0, "transition_type": "crossfade", "transition_seconds": 0.5,
        "speed_curve": "linear", "intro_enabled": False, "outro_enabled": False,
        "seed": 0, "export_png": True,
        "fitness_mode": "balanced",
        "camera_mode": "cinematic",
        "temporal_smooth": True, "smooth_strength": 0.3,
        "temporal_coherence": 0.3, "inheritance_strength": 0.8,
        "progressive_shapes": False, "motion_blur": False,
        "motion_blur_strength": 0.3, "keyframe_interval": 1,
        "use_residual_guidance": True, "use_multiscale_fitness": False,
        "use_annealing": True, "use_de_blend": True,
        "use_shape_split": True, "allow_dynamic_shape_count": False,
        "use_fitness_cache": True, "use_diversity_injection": True,
        "use_importance_tracking": True, "use_gradient_mutation": True,
        "evaluation_scale": 0.5, "enable_roi": False, "roi_weight": 1.5,
        "quality_tier": "hd", "generate_per_image_mp4": True,
        "generate_thumbnail": True, "post_sharpen": 0.3,
        "post_color_grade": "none", "post_vignette": 0.2,
        "post_film_grain": 0.0, "caption_style": "none", "audio_mood": "ambient",
    },
    "High Quality": {
        "canvas_w": 256, "canvas_h": 256, "num_shapes": 150,
        "population_size": 120, "mutation_rate": 0.015, "elite_count": 6,
        "alpha": 100, "generations": 2000, "segment_duration": 10.0, "fps": 24,
        "display_w": 640, "display_h": 480, "n_sides": 5,
        "shape_mode": "mixed", "use_smart_init": True,
        "use_hill_climb": True, "hill_climb_steps": 40,
        "use_adaptive_mutation": True, "use_ssim": True,
        "use_multiscale": True, "multiscale_ratio": 0.3, "color_space": "hsv",
        "stop_mode": "target_mse", "target_mse": 0.005,
        "convergence_threshold": 5e-6, "convergence_patience": 40,
        "hold_seconds": 2.0, "transition_type": "fade_black", "transition_seconds": 1.0,
        "speed_curve": "ease_in_out", "intro_enabled": True, "outro_enabled": True,
        "seed": 0, "export_png": True,
        "fitness_mode": "cinematic",
        "camera_mode": "cinematic",
        "temporal_smooth": True, "smooth_strength": 0.25,
        "temporal_coherence": 0.4, "inheritance_strength": 0.9,
        "progressive_shapes": True, "motion_blur": True,
        "motion_blur_strength": 0.25, "keyframe_interval": 1,
        "use_residual_guidance": True, "use_multiscale_fitness": True,
        "use_annealing": True, "use_de_blend": True,
        "use_shape_split": True, "allow_dynamic_shape_count": True,
        "use_fitness_cache": True, "use_diversity_injection": True,
        "use_importance_tracking": True, "use_gradient_mutation": True,
        "evaluation_scale": 1.0, "enable_roi": False, "roi_weight": 1.5,
        "quality_tier": "hd", "generate_per_image_mp4": True,
        "generate_thumbnail": True, "post_sharpen": 0.4,
        "post_color_grade": "cinematic", "post_vignette": 0.3,
        "post_film_grain": 0.02, "caption_style": "cinematic", "audio_mood": "hopeful",
    },
    "Cinematic Smooth": {
        "canvas_w": 256, "canvas_h": 256, "num_shapes": 120,
        "population_size": 100, "mutation_rate": 0.018, "elite_count": 5,
        "alpha": 110, "generations": 1000, "segment_duration": 10.0, "fps": 24,
        "display_w": 640, "display_h": 480, "n_sides": 5,
        "shape_mode": "mixed", "use_smart_init": True,
        "use_hill_climb": True, "hill_climb_steps": 25,
        "use_adaptive_mutation": True, "use_ssim": True,
        "use_multiscale": False, "multiscale_ratio": 0.3, "color_space": "hsv",
        "stop_mode": "target_mse", "target_mse": 0.008,
        "convergence_threshold": 1e-5, "convergence_patience": 30,
        "hold_seconds": 1.5, "transition_type": "crossfade", "transition_seconds": 0.8,
        "speed_curve": "ease_in_out", "intro_enabled": False, "outro_enabled": False,
        "seed": 0, "export_png": True,
        "fitness_mode": "cinematic",
        "camera_mode": "cinematic",
        "temporal_smooth": True, "smooth_strength": 0.25,
        "temporal_coherence": 0.5, "inheritance_strength": 0.9,
        "progressive_shapes": True, "motion_blur": True,
        "motion_blur_strength": 0.25, "keyframe_interval": 1,
        "use_residual_guidance": True, "use_multiscale_fitness": False,
        "use_annealing": True, "use_de_blend": True,
        "use_shape_split": True, "allow_dynamic_shape_count": False,
        "use_fitness_cache": True, "use_diversity_injection": True,
        "use_importance_tracking": True, "use_gradient_mutation": True,
        "evaluation_scale": 1.0, "enable_roi": False, "roi_weight": 1.5,
        "quality_tier": "hd", "generate_per_image_mp4": True,
        "generate_thumbnail": True, "post_sharpen": 0.5,
        "post_color_grade": "cinematic", "post_vignette": 0.35,
        "post_film_grain": 0.03, "caption_style": "cinematic", "audio_mood": "melancholic",
    },
    "Artistic HSV": {
        "canvas_w": 192, "canvas_h": 192, "num_shapes": 100,
        "population_size": 80, "mutation_rate": 0.02, "elite_count": 4,
        "alpha": 90, "generations": 1500, "segment_duration": 8.0, "fps": 20,
        "display_w": 480, "display_h": 360, "n_sides": 5,
        "shape_mode": "ellipse", "use_smart_init": True,
        "use_hill_climb": True, "hill_climb_steps": 25,
        "use_adaptive_mutation": True, "use_ssim": False,
        "use_multiscale": False, "multiscale_ratio": 0.3, "color_space": "hsv",
        "stop_mode": "target_mse", "target_mse": 0.01,
        "convergence_threshold": 1e-5, "convergence_patience": 30,
        "hold_seconds": 1.5, "transition_type": "crossfade", "transition_seconds": 0.8,
        "speed_curve": "ease_out", "intro_enabled": False, "outro_enabled": False,
        "seed": 0, "export_png": True,
        "fitness_mode": "experimental",
        "camera_mode": "dynamic",
        "temporal_smooth": True, "smooth_strength": 0.35,
        "temporal_coherence": 0.3, "inheritance_strength": 0.8,
        "progressive_shapes": False, "motion_blur": False,
        "motion_blur_strength": 0.3, "keyframe_interval": 1,
        "use_residual_guidance": True, "use_multiscale_fitness": False,
        "use_annealing": True, "use_de_blend": True,
        "use_shape_split": True, "allow_dynamic_shape_count": True,
        "use_fitness_cache": True, "use_diversity_injection": True,
        "use_importance_tracking": True, "use_gradient_mutation": True,
        "evaluation_scale": 0.5, "enable_roi": False, "roi_weight": 1.5,
        "quality_tier": "hd", "generate_per_image_mp4": True,
        "generate_thumbnail": True, "post_sharpen": 0.3,
        "post_color_grade": "none", "post_vignette": 0.2,
        "post_film_grain": 0.0, "caption_style": "none", "audio_mood": "mysterious",
    },
}


# ---------------------------------------------------------------------------
# Color space utilities
# ---------------------------------------------------------------------------
class ColorSpace:
    @staticmethod
    def rgb_to_hsv(r, g, b):
        h, s, v = colorsys.rgb_to_hsv(r, g, b); return (h * 360.0, s, v)
    @staticmethod
    def hsv_to_rgb(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v); return (r, g, b)
    @staticmethod
    def sample_color_hsv(target_small, y, x):
        r, g, b = target_small[int(y), int(x)]; return ColorSpace.rgb_to_hsv(r, g, b)


# ---------------------------------------------------------------------------
# Shape system (18 shapes + mixed)
# ---------------------------------------------------------------------------
SHAPE_KINDS = [
    "triangle", "quad", "ngon", "ellipse", "bezier_q", "bezier_c", "line", "star",
    "rectangle", "circle", "ring", "rounded_rect",
    "gradient_rect", "gradient_ellipse",
    "spline", "cross", "heart", "arrow"
]
VERTEX_KINDS = {"triangle", "quad", "ngon", "bezier_q", "bezier_c", "line", "spline"}


class ShapeFactory:
    @staticmethod
    def random_kind(): return random.choice(SHAPE_KINDS)

    @staticmethod
    def random_shape(kind, canvas_size, target_small=None, edge_prob=None,
                     alpha=128, n_sides=5, color_space="rgb"):
        w, h = canvas_size
        if color_space == "hsv":
            color = (random.uniform(0, 360), random.random(), random.random(), alpha / 255.0)
        else:
            color = (random.random(), random.random(), random.random(), alpha / 255.0)
        if edge_prob is not None and random.random() < 0.7:
            flat_idx = np.random.choice(edge_prob.size, p=edge_prob.flatten())
            cy, cx = np.unravel_index(flat_idx, edge_prob.shape)
            cx, cy = float(cx), float(cy)
            if target_small is not None:
                iy = int(np.clip(cy, 0, target_small.shape[0]-1))
                ix = int(np.clip(cx, 0, target_small.shape[1]-1))
                if color_space == "hsv":
                    color = ColorSpace.sample_color_hsv(target_small, iy, ix) + (alpha / 255.0,)
                else:
                    color = tuple(target_small[iy, ix]) + (alpha / 255.0,)
        else:
            cx, cy = random.random() * w, random.random() * h
        if kind == "triangle":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, 3), "color": color}
        if kind == "quad":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, 4), "color": color}
        if kind == "ngon":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, n_sides), "color": color}
        if kind == "ellipse":
            return {"kind": kind, "cx": cx, "cy": cy, "rx": random.uniform(5, w*0.3),
                    "ry": random.uniform(5, h*0.3), "rot": random.uniform(0, 180), "color": color}
        if kind == "bezier_q":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, 3, 0.5),
                    "color": color, "thickness": random.uniform(1, 6)}
        if kind == "bezier_c":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, 4, 0.5),
                    "color": color, "thickness": random.uniform(1, 6)}
        if kind == "line":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, 2, 0.6),
                    "color": color, "thickness": random.uniform(1, 8)}
        if kind == "star":
            outer_r = random.uniform(5, min(w, h) * 0.3)
            return {"kind": kind, "cx": cx, "cy": cy, "outer_r": outer_r,
                    "inner_r": outer_r * random.uniform(0.3, 0.7),
                    "points": random.randint(4, 8), "rot": random.uniform(0, 360), "color": color}
        if kind == "rectangle":
            rw, rh = random.uniform(10, w*0.4), random.uniform(10, h*0.4)
            return {"kind": kind, "x": cx-rw/2, "y": cy-rh/2, "w": rw, "h": rh, "color": color}
        if kind == "circle":
            return {"kind": kind, "cx": cx, "cy": cy, "r": random.uniform(5, min(w,h)*0.3), "color": color}
        if kind == "ring":
            r = random.uniform(10, min(w,h)*0.3)
            return {"kind": kind, "cx": cx, "cy": cy, "r": r,
                    "thickness": random.uniform(2, r*0.5), "color": color}
        if kind == "rounded_rect":
            rw, rh = random.uniform(10, w*0.4), random.uniform(10, h*0.4)
            return {"kind": kind, "x": cx-rw/2, "y": cy-rh/2, "w": rw, "h": rh,
                    "corner_r": random.uniform(2, min(rw,rh)*0.4), "color": color}
        if kind == "gradient_rect":
            rw, rh = random.uniform(10, w*0.4), random.uniform(10, h*0.4)
            c2 = (random.uniform(0,360), random.random(), random.random(), alpha/255.0) if color_space == "hsv" \
                 else (random.random(), random.random(), random.random(), alpha/255.0)
            return {"kind": kind, "x": cx-rw/2, "y": cy-rh/2, "w": rw, "h": rh,
                    "color": color, "color2": c2, "angle": random.uniform(0, 360)}
        if kind == "gradient_ellipse":
            c2 = (random.uniform(0,360), random.random(), random.random(), alpha/255.0) if color_space == "hsv" \
                 else (random.random(), random.random(), random.random(), alpha/255.0)
            return {"kind": kind, "cx": cx, "cy": cy, "rx": random.uniform(5, w*0.3),
                    "ry": random.uniform(5, h*0.3), "color": color, "color2": c2}
        if kind == "spline":
            return {"kind": kind, "verts": ShapeFactory._verts_around(cx, cy, w, h, random.randint(3,6), 0.5),
                    "color": color, "thickness": random.uniform(1, 6)}
        if kind == "cross":
            sz = random.uniform(10, min(w,h)*0.3)
            return {"kind": kind, "cx": cx, "cy": cy, "size": sz,
                    "thickness": sz * random.uniform(0.2, 0.4), "color": color}
        if kind == "heart":
            return {"kind": kind, "cx": cx, "cy": cy, "size": random.uniform(10, min(w,h)*0.3), "color": color}
        if kind == "arrow":
            return {"kind": kind, "cx": cx, "cy": cy, "size": random.uniform(10, min(w,h)*0.3),
                    "angle": random.uniform(0, 360), "color": color}
        raise ValueError(f"Unknown shape: {kind}")

    @staticmethod
    def _verts_around(cx, cy, w, h, n, scale=0.3):
        return [(float(np.clip(cx+(random.random()-0.5)*w*scale, 0, w-1)),
                 float(np.clip(cy+(random.random()-0.5)*h*scale, 0, h-1))) for _ in range(n)]

    @staticmethod
    def deep_copy(shape):
        s = {"kind": shape["kind"]}
        for k, v in shape.items():
            if k == "verts": s["verts"] = [(x, y) for x, y in v]
            elif k in ("color", "color2"): s[k] = tuple(v)
            else: s[k] = v
        return s

    @staticmethod
    def mutate(shape, canvas_size, mutation_rate, n_sides=5, allow_kind_change=False, color_space="rgb"):
        s = ShapeFactory.deep_copy(shape)
        w, h = canvas_size; scale = mutation_rate * 4
        if allow_kind_change and random.random() < 0.02:
            return ShapeFactory.random_shape(ShapeFactory.random_kind(), canvas_size,
                                             alpha=int(s["color"][3]*255), n_sides=n_sides, color_space=color_space)
        kind = s["kind"]; action = random.random()
        if action < 0.15:
            if color_space == "hsv":
                hv, sv, vv, a = s["color"]
                hv = (hv + (random.random()-0.5) * mutation_rate * 60) % 360
                sv = np.clip(sv + (random.random()-0.5) * mutation_rate * 0.4, 0, 1)
                vv = np.clip(vv + (random.random()-0.5) * mutation_rate * 0.4, 0, 1)
                a = np.clip(a + (random.random()-0.5) * mutation_rate * 0.6, 0.05, 1)
                s["color"] = (hv, sv, vv, a)
                if "color2" in s:
                    h2, s2, v2, a2 = s["color2"]
                    h2 = (h2 + (random.random()-0.5) * mutation_rate * 60) % 360
                    s2 = np.clip(s2 + (random.random()-0.5) * mutation_rate * 0.4, 0, 1)
                    v2 = np.clip(v2 + (random.random()-0.5) * mutation_rate * 0.4, 0, 1)
                    s["color2"] = (h2, s2, v2, a2)
            else:
                r, g, b, a = s["color"]; d = mutation_rate * 2
                s["color"] = (np.clip(r+(random.random()-0.5)*d,0,1),
                              np.clip(g+(random.random()-0.5)*d,0,1),
                              np.clip(b+(random.random()-0.5)*d,0,1),
                              np.clip(a+(random.random()-0.5)*mutation_rate*0.6,0.05,1))
                if "color2" in s:
                    r2,g2,b2,a2 = s["color2"]
                    s["color2"] = (np.clip(r2+(random.random()-0.5)*d,0,1),
                                   np.clip(g2+(random.random()-0.5)*d,0,1),
                                   np.clip(b2+(random.random()-0.5)*d,0,1), a2)
            return s
        if kind in VERTEX_KINDS:
            idx = random.randrange(len(s["verts"]))
            vx, vy = s["verts"][idx]
            s["verts"][idx] = (np.clip(vx+(random.random()-0.5)*w*scale,0,w-1),
                               np.clip(vy+(random.random()-0.5)*h*scale,0,h-1))
            if "thickness" in s and random.random() < 0.3:
                s["thickness"] = np.clip(s["thickness"]+(random.random()-0.5)*2, 1, 15)
        elif kind in ("ellipse", "gradient_ellipse"):
            q = random.random()
            if q < 0.4:
                s["cx"] = np.clip(s["cx"]+(random.random()-0.5)*w*scale,0,w)
                s["cy"] = np.clip(s["cy"]+(random.random()-0.5)*h*scale,0,h)
            else:
                s["rx"] = np.clip(s["rx"]*(1+(random.random()-0.5)*scale),2,w)
                s["ry"] = np.clip(s["ry"]*(1+(random.random()-0.5)*scale),2,h)
            if "rot" in s: s["rot"] = (s["rot"]+(random.random()-0.5)*30*scale) % 180
        elif kind in ("star", "circle", "ring", "cross", "heart", "arrow"):
            q = random.random()
            if q < 0.4:
                s["cx"] = np.clip(s["cx"]+(random.random()-0.5)*w*scale,0,w)
                s["cy"] = np.clip(s["cy"]+(random.random()-0.5)*h*scale,0,h)
            elif q < 0.7:
                for k in ("outer_r","inner_r","r","size"):
                    if k in s: s[k] = np.clip(s[k]*(1+(random.random()-0.5)*scale), 2, min(w,h))
            else:
                for k in ("rot","angle"):
                    if k in s: s[k] = (s[k]+(random.random()-0.5)*30*scale) % 360
                if "thickness" in s and kind == "ring":
                    s["thickness"] = np.clip(s["thickness"]*(1+(random.random()-0.5)*scale), 1, s.get("r",10))
        elif kind in ("rectangle", "rounded_rect", "gradient_rect"):
            q = random.random()
            if q < 0.4:
                s["x"] = np.clip(s["x"]+(random.random()-0.5)*w*scale,0,w)
                s["y"] = np.clip(s["y"]+(random.random()-0.5)*h*scale,0,h)
            elif q < 0.7:
                s["w"] = np.clip(s["w"]*(1+(random.random()-0.5)*scale),5,w)
                s["h"] = np.clip(s["h"]*(1+(random.random()-0.5)*scale),5,h)
            else:
                if "corner_r" in s: s["corner_r"] = np.clip(s["corner_r"]*(1+(random.random()-0.5)*scale),1,min(s["w"],s["h"])*0.5)
                if "angle" in s: s["angle"] = (s["angle"]+(random.random()-0.5)*30*scale) % 360
        return s

    @staticmethod
    def micro_mutate(shape, canvas_size, color_space="rgb"):
        s = ShapeFactory.deep_copy(shape); w, h = canvas_size; kind = s["kind"]
        if kind in VERTEX_KINDS:
            idx = random.randrange(len(s["verts"]))
            vx, vy = s["verts"][idx]
            s["verts"][idx] = (np.clip(vx+(random.random()-0.5)*4,0,w-1),
                               np.clip(vy+(random.random()-0.5)*4,0,h-1))
        elif kind in ("ellipse", "gradient_ellipse"):
            if random.random() < 0.5:
                s["cx"] = np.clip(s["cx"]+(random.random()-0.5)*3,0,w)
                s["cy"] = np.clip(s["cy"]+(random.random()-0.5)*3,0,h)
            else:
                s["rx"] = np.clip(s["rx"]+(random.random()-0.5)*2,2,w)
                s["ry"] = np.clip(s["ry"]+(random.random()-0.5)*2,2,h)
        elif kind in ("star","circle","ring","cross","heart","arrow"):
            s["cx"] = np.clip(s["cx"]+(random.random()-0.5)*3,0,w)
            s["cy"] = np.clip(s["cy"]+(random.random()-0.5)*3,0,h)
        elif kind in ("rectangle","rounded_rect","gradient_rect"):
            s["x"] = np.clip(s["x"]+(random.random()-0.5)*3,0,w)
            s["y"] = np.clip(s["y"]+(random.random()-0.5)*3,0,h)
        return s

    @staticmethod
    def render(draw, shape, color_space="rgb"):
        kind = shape["kind"]
        if color_space == "hsv":
            r, g, b = ColorSpace.hsv_to_rgb(shape["color"][0], shape["color"][1], shape["color"][2])
            a = shape["color"][3]
        else:
            r, g, b, a = shape["color"]
        fill = (int(r*255), int(g*255), int(b*255), int(a*255))
        if kind in ("triangle","quad","ngon"):
            draw.polygon([(int(x),int(y)) for x,y in shape["verts"]], fill=fill)
        elif kind == "ellipse":
            cx,cy,rx,ry,rot = shape["cx"],shape["cy"],shape["rx"],shape["ry"],shape["rot"]
            if abs(rot)<0.5 or abs(rx-ry)<1:
                draw.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], fill=fill)
            else:
                draw.polygon(ShapeFactory._rotated_ellipse_points(cx,cy,rx,ry,rot,32), fill=fill)
        elif kind == "bezier_q":
            p0,p1,p2 = shape["verts"]
            pts = [(int((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t**2*p2[0]),
                    int((1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t**2*p2[1])) for t in np.linspace(0,1,40)]
            draw.line(pts, fill=fill, width=max(1,int(shape["thickness"])))
        elif kind == "bezier_c":
            p0,p1,p2,p3 = shape["verts"]
            pts = [(int((1-t)**3*p0[0]+3*(1-t)**2*t*p1[0]+3*(1-t)*t**2*p2[0]+t**3*p3[0]),
                    int((1-t)**3*p0[1]+3*(1-t)**2*t*p1[1]+3*(1-t)*t**2*p2[1]+t**3*p3[1])) for t in np.linspace(0,1,50)]
            draw.line(pts, fill=fill, width=max(1,int(shape["thickness"])))
        elif kind == "line":
            p0,p1 = shape["verts"]
            draw.line([(int(p0[0]),int(p0[1])),(int(p1[0]),int(p1[1]))], fill=fill, width=max(1,int(shape["thickness"])))
        elif kind == "star":
            cx,cy = shape["cx"],shape["cy"]
            pts = [(int(cx+(shape["outer_r"] if i%2==0 else shape["inner_r"])*np.cos(np.deg2rad(shape["rot"]+i*180/shape["points"]))),
                    int(cy+(shape["outer_r"] if i%2==0 else shape["inner_r"])*np.sin(np.deg2rad(shape["rot"]+i*180/shape["points"]))))
                   for i in range(shape["points"]*2)]
            draw.polygon(pts, fill=fill)
        elif kind == "rectangle":
            draw.rectangle([shape["x"],shape["y"],shape["x"]+shape["w"],shape["y"]+shape["h"]], fill=fill)
        elif kind == "circle":
            draw.ellipse([shape["cx"]-shape["r"],shape["cy"]-shape["r"],shape["cx"]+shape["r"],shape["cy"]+shape["r"]], fill=fill)
        elif kind == "ring":
            draw.ellipse([shape["cx"]-shape["r"],shape["cy"]-shape["r"],shape["cx"]+shape["r"],shape["cy"]+shape["r"]], fill=fill)
            ir = shape["r"]-shape["thickness"]
            if ir > 0: draw.ellipse([shape["cx"]-ir,shape["cy"]-ir,shape["cx"]+ir,shape["cy"]+ir], fill=(0,0,0,0))
        elif kind == "rounded_rect":
            draw.rounded_rectangle([shape["x"],shape["y"],shape["x"]+shape["w"],shape["y"]+shape["h"]], radius=shape["corner_r"], fill=fill)
        elif kind == "gradient_rect":
            if color_space=="hsv":
                r2,g2,b2 = ColorSpace.hsv_to_rgb(shape["color2"][0],shape["color2"][1],shape["color2"][2]); a2=shape["color2"][3]
            else: r2,g2,b2,a2 = shape["color2"]
            for i in range(20):
                t=i/19; cf=(int((r+(r2-r)*t)*255),int((g+(g2-g)*t)*255),int((b+(b2-b)*t)*255),int((a+(a2-a)*t)*255))
                sw=shape["w"]/20; draw.rectangle([shape["x"]+i*sw,shape["y"],shape["x"]+(i+1)*sw,shape["y"]+shape["h"]], fill=cf)
        elif kind == "gradient_ellipse":
            if color_space=="hsv":
                r2,g2,b2 = ColorSpace.hsv_to_rgb(shape["color2"][0],shape["color2"][1],shape["color2"][2]); a2=shape["color2"][3]
            else: r2,g2,b2,a2 = shape["color2"]
            for i in range(15,0,-1):
                t=i/15; cf=(int((r+(r2-r)*(1-t))*255),int((g+(g2-g)*(1-t))*255),int((b+(b2-b)*(1-t))*255),int((a+(a2-a)*(1-t))*255))
                draw.ellipse([shape["cx"]-shape["rx"]*t,shape["cy"]-shape["ry"]*t,shape["cx"]+shape["rx"]*t,shape["cy"]+shape["ry"]*t], fill=cf)
        elif kind == "spline":
            pts_raw = shape["verts"]
            if len(pts_raw)<2: return
            pts = []
            for i in range(len(pts_raw)-1):
                p0=pts_raw[max(0,i-1)]; p1=pts_raw[i]; p2=pts_raw[min(len(pts_raw)-1,i+1)]; p3=pts_raw[min(len(pts_raw)-1,i+2)]
                for t in np.linspace(0,1,10,endpoint=False):
                    t2=t*t; t3=t2*t
                    pts.append((int(0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)),
                                int(0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3))))
            if pts: draw.line(pts, fill=fill, width=max(1,int(shape["thickness"])))
        elif kind == "cross":
            cx,cy,sz,th = shape["cx"],shape["cy"],shape["size"],shape["thickness"]
            draw.rectangle([cx-sz,cy-th/2,cx+sz,cy+th/2], fill=fill)
            draw.rectangle([cx-th/2,cy-sz,cx+th/2,cy+sz], fill=fill)
        elif kind == "heart":
            cx,cy,sz = shape["cx"],shape["cy"],shape["size"]
            pts = [(int(cx+16*np.sin(t)**3*sz/16), int(cy-(13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))*sz/16))
                   for t in np.linspace(0,2*np.pi,50)]
            draw.polygon(pts, fill=fill)
        elif kind == "arrow":
            cx,cy,sz = shape["cx"],shape["cy"],shape["size"]
            ang = np.deg2rad(shape["angle"]); ca,sa = np.cos(ang),np.sin(ang)
            raw = [(sz,0),(sz*0.6,-sz*0.3),(sz*0.6,-sz*0.15),(-sz,-sz*0.15),(-sz,sz*0.15),(sz*0.6,sz*0.15),(sz*0.6,sz*0.3)]
            draw.polygon([(int(px*ca-py*sa+cx),int(px*sa+py*ca+cy)) for px,py in raw], fill=fill)

    @staticmethod
    def _rotated_ellipse_points(cx,cy,rx,ry,rot_deg,n=32):
        rot=np.deg2rad(rot_deg); cr,sr=np.cos(rot),np.sin(rot)
        return [(int(cx+rx*np.cos(t)*cr-ry*np.sin(t)*sr), int(cy+rx*np.cos(t)*sr+ry*np.sin(t)*cr))
                for t in [2*np.pi*i/n for i in range(n)]]

    @staticmethod
    def upscale(shape, factor):
        s = ShapeFactory.deep_copy(shape); k = s["kind"]
        if k in VERTEX_KINDS:
            s["verts"] = [(x*factor,y*factor) for x,y in s["verts"]]
            if "thickness" in s: s["thickness"] *= factor
        elif k in ("ellipse","circle","ring","gradient_ellipse"):
            for j in ("cx","cy","rx","ry","r"):
                if j in s: s[j] *= factor
        elif k in ("star","cross","heart","arrow"):
            for j in ("cx","cy","outer_r","inner_r","size"):
                if j in s: s[j] *= factor
        elif k in ("rectangle","rounded_rect","gradient_rect"):
            for j in ("x","y","w","h","corner_r"):
                if j in s: s[j] *= factor
        return s


# ---------------------------------------------------------------------------
# Algorithm components
# ---------------------------------------------------------------------------
class ResidualMap:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
        self.error_map = None
        self.placement_prob = None
        self.update_rate = 0.3

    def update(self, rendered, target):
        diff = np.abs(rendered - target)
        new_error = np.mean(diff, axis=2)
        error_img = Image.fromarray((new_error * 255).astype(np.uint8))
        error_img = error_img.filter(ImageFilter.GaussianBlur(radius=3))
        new_error = np.asarray(error_img).astype(np.float32) / 255.0
        if self.error_map is None:
            self.error_map = new_error
        else:
            self.error_map = (self.update_rate * new_error +
                              (1 - self.update_rate) * self.error_map)
        eps = 0.001
        self.placement_prob = (self.error_map + eps)
        self.placement_prob /= self.placement_prob.sum()

    def sample_position(self):
        h, w = self.canvas_size
        flat_idx = np.random.choice(self.placement_prob.size,
                                     p=self.placement_prob.flatten())
        cy, cx = np.unravel_index(flat_idx, self.placement_prob.shape)
        return float(cx), float(cy)

    def get_error_at(self, x, y):
        if self.error_map is None: return 0.5
        ix = int(np.clip(x, 0, self.canvas_size[0] - 1))
        iy = int(np.clip(y, 0, self.canvas_size[1] - 1))
        return float(self.error_map[iy, ix])


class MultiScaleFitness:
    def __init__(self, use_ssim=False, scales=(1.0, 0.5, 0.25), weights=(0.5, 0.3, 0.2)):
        self.use_ssim = use_ssim
        self.scales = scales
        self.weights = weights
        self._target_cache = {}

    def _get_target_at_scale(self, target_small, scale):
        if scale == 1.0: return target_small
        key = scale
        if key not in self._target_cache:
            h, w = target_small.shape[:2]
            new_size = (max(8, int(w * scale)), max(8, int(h * scale)))
            img = Image.fromarray((target_small * 255).astype(np.uint8))
            img = img.resize(new_size, Image.BILINEAR)
            self._target_cache[key] = np.asarray(img).astype(np.float32) / 255.0
        return self._target_cache[key]

    def _ssim(self, img1, img2):
        C1, C2 = 0.01**2, 0.03**2
        def blur(a):
            return np.array(Image.fromarray((a*255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)/255
        m1, m2 = blur(img1), blur(img2)
        s1 = np.maximum(blur(img1*img1)-m1*m1, 0)
        s2 = np.maximum(blur(img2*img2)-m2*m2, 0)
        s12 = blur(img1*img2) - m1*m2
        return float(np.mean((2*m1*m2+C1)*(2*s12+C2) /
                             ((m1*m1+m2*m2+C1)*(s1+s2+C2))))

    def evaluate(self, rendered, target_small):
        total = 0.0
        for scale, weight in zip(self.scales, self.weights):
            target = self._get_target_at_scale(target_small, scale)
            rend = rendered
            if scale != 1.0:
                h, w = rendered.shape[:2]
                new_size = (max(8, int(w * scale)), max(8, int(h * scale)))
                img = Image.fromarray((rendered * 255).astype(np.uint8))
                img = img.resize(new_size, Image.BILINEAR)
                rend = np.asarray(img).astype(np.float32) / 255.0
            if rend.shape != target.shape:
                t_img = Image.fromarray((target * 255).astype(np.uint8))
                t_img = t_img.resize((rend.shape[1], rend.shape[0]), Image.BILINEAR)
                target = np.asarray(t_img).astype(np.float32) / 255.0
            if self.use_ssim:
                score = 1.0 - self._ssim(rend, target)
            else:
                score = float(np.mean((rend.flatten() - target.flatten()) ** 2))
            total += weight * score
        return total


class AnnealingAcceptor:
    def __init__(self, initial_temp=0.05, cooling_rate=0.995, min_temp=0.0001):
        self.temp = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def accept(self, current_fitness, new_fitness):
        if new_fitness <= current_fitness: return True
        delta = new_fitness - current_fitness
        prob = np.exp(-delta / max(self.min_temp, self.temp))
        return random.random() < prob

    def cool(self):
        self.temp = max(self.min_temp, self.temp * self.cooling_rate)

    def reset(self):
        self.temp = self.initial_temp


class DEBlender:
    @staticmethod
    def blend_shapes(shape_a, shape_b, shape_target, F=0.5, CR=0.7):
        child = ShapeFactory.deep_copy(shape_target)
        if "verts" in child and "verts" in shape_a and "verts" in shape_b:
            n_verts = min(len(child["verts"]), len(shape_a["verts"]), len(shape_b["verts"]))
            for i in range(n_verts):
                if random.random() < CR:
                    ax, ay = shape_a["verts"][i]
                    bx, by = shape_b["verts"][i]
                    tx, ty = child["verts"][i]
                    new_x = tx + F * (ax - bx)
                    new_y = ty + F * (ay - by)
                    child["verts"][i] = (new_x, new_y)
        for param in ("cx", "cy", "rx", "ry", "r", "x", "y", "w", "h",
                      "size", "outer_r", "inner_r", "rot", "angle",
                      "thickness", "corner_r"):
            if param in child and param in shape_a and param in shape_b:
                if random.random() < CR:
                    child[param] = child[param] + F * (shape_a[param] - shape_b[param])
        if "color" in child and "color" in shape_a and "color" in shape_b:
            color_list = list(child["color"])
            for i in range(len(color_list)):
                if random.random() < CR:
                    color_list[i] = color_list[i] + F * (shape_a["color"][i] - shape_b["color"][i])
            child["color"] = tuple(color_list)
        return child


class FitnessCache:
    def __init__(self, max_size=5000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_individual(self, individual):
        parts = []
        for shape in individual:
            h_parts = [shape["kind"]]
            for k in sorted(shape.keys()):
                if k == "kind": continue
                v = shape[k]
                if k == "verts":
                    h_parts.append(tuple((round(x, 4), round(y, 4)) for x, y in v))
                elif k in ("color", "color2"):
                    h_parts.append(tuple(round(c, 4) for c in v))
                elif isinstance(v, (int, float)):
                    h_parts.append(round(v, 4))
            parts.append(tuple(h_parts))
        return hash(tuple(parts))

    def get(self, individual):
        h = self._hash_individual(individual)
        if h in self.cache:
            self.hits += 1; return self.cache[h]
        self.misses += 1; return None

    def put(self, individual, fitness):
        if len(self.cache) >= self.max_size:
            keys = list(self.cache.keys())
            for k in keys[:len(keys) // 4]:
                del self.cache[k]
        h = self._hash_individual(individual)
        self.cache[h] = fitness


class DiversityMonitor:
    def __init__(self, min_diversity=0.05, check_interval=20, inject_count=4):
        self.min_diversity = min_diversity
        self.check_interval = check_interval
        self.inject_count = inject_count
        self.diversity_history = []

    def compute_diversity(self, population):
        if len(population) < 2: return 1.0
        sample_size = min(20, len(population))
        sample = random.sample(population, sample_size)
        total_dist = 0.0; pairs = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                c1 = sample[i][0]["color"][:3]
                c2 = sample[j][0]["color"][:3]
                dist = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
                total_dist += dist; pairs += 1
        avg_dist = total_dist / max(1, pairs)
        self.diversity_history.append(avg_dist)
        return avg_dist

    def should_inject(self, generation):
        if generation % self.check_interval != 0: return False
        if len(self.diversity_history) < 3: return False
        recent_avg = np.mean(self.diversity_history[-3:])
        return recent_avg < self.min_diversity


class ShapeImportanceTracker:
    def __init__(self):
        self.importance_scores = {}

    def compute_importance(self, individual, fitness_fn, n_samples=10):
        base_fitness = fitness_fn(individual)
        self.importance_scores = {}
        indices = list(range(len(individual)))
        if len(indices) > n_samples:
            indices = random.sample(indices, n_samples)
        for idx in indices:
            modified = [ShapeFactory.deep_copy(s) for s in individual]
            color = list(modified[idx]["color"])
            color[-1] = 0.0
            modified[idx]["color"] = tuple(color)
            mod_fitness = fitness_fn(modified)
            importance = mod_fitness - base_fitness
            self.importance_scores[idx] = max(0.0, importance)
        if self.importance_scores:
            max_imp = max(self.importance_scores.values())
            if max_imp > 0:
                self.importance_scores = {k: v / max_imp for k, v in self.importance_scores.items()}

    def get_importance(self, idx):
        return self.importance_scores.get(idx, 0.5)

    def get_most_important(self, n=1):
        if not self.importance_scores: return []
        sorted_shapes = sorted(self.importance_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_shapes[:n]]


# ---------------------------------------------------------------------------
# Temporal engine (frame-to-frame coherence)
# ---------------------------------------------------------------------------
class TemporalEngine:
    def __init__(self, params):
        self.previous_individual = None
        self.previous_render = None
        self.coherence_weight = float(params.get("temporal_coherence", 0.3))
        self.inheritance_strength = float(params.get("inheritance_strength", 0.8))
        self.progressive_shapes = params.get("progressive_shapes", False)
        self.keyframe_interval = int(params.get("keyframe_interval", 1))
        self.motion_blur = params.get("motion_blur", False)
        self.motion_blur_strength = float(params.get("motion_blur_strength", 0.3))
        self.frame_count = 0
        self.max_shapes_target = int(params.get("num_shapes", 80))

    def reset(self):
        self.previous_individual = None
        self.previous_render = None
        self.frame_count = 0

    def get_target_shape_count(self):
        if not self.progressive_shapes:
            return self.max_shapes_target
        progress = min(1.0, self.frame_count / 100.0)
        ratio = 0.2 + 0.8 * (3 * progress**2 - 2 * progress**3)
        return max(10, int(self.max_shapes_target * ratio))

    def create_seeded_population(self, engine, population_size):
        if self.previous_individual is None:
            return None
        seeded_pop = []
        n_inherited = int(population_size * self.inheritance_strength)
        for i in range(n_inherited):
            inherited = [ShapeFactory.deep_copy(s) for s in self.previous_individual]
            target_count = self.get_target_shape_count()
            current_count = len(inherited)
            if current_count < target_count:
                for _ in range(target_count - current_count):
                    new_shape = ShapeFactory.random_shape(
                        ShapeFactory.random_kind(), engine.canvas_size,
                        alpha=engine.alpha, n_sides=engine.n_sides,
                        color_space=engine.color_space)
                    inherited.append(new_shape)
            elif current_count > target_count:
                inherited = inherited[:target_count]
            n_mutations = max(1, len(inherited) // 10)
            for _ in range(n_mutations):
                idx = random.randrange(len(inherited))
                inherited[idx] = ShapeFactory.micro_mutate(
                    inherited[idx], engine.canvas_size, engine.color_space)
            seeded_pop.append(inherited)
        while len(seeded_pop) < population_size:
            count = self.get_target_shape_count()
            seeded_pop.append([ShapeFactory.random_shape(
                ShapeFactory.random_kind(), engine.canvas_size,
                alpha=engine.alpha, n_sides=engine.n_sides,
                color_space=engine.color_space) for _ in range(count)])
        return seeded_pop

    def temporal_coherence_penalty(self, individual, previous_individual):
        if previous_individual is None or len(individual) != len(previous_individual):
            return 0.0
        total_penalty = 0.0; n_compared = 0
        for curr, prev in zip(individual, previous_individual):
            if curr["kind"] != prev["kind"]:
                total_penalty += 0.5; n_compared += 1; continue
            curr_cx, curr_cy = self._get_center(curr)
            prev_cx, prev_cy = self._get_center(prev)
            dx = curr_cx - prev_cx; dy = curr_cy - prev_cy
            distance = (dx**2 + dy**2) ** 0.5
            if distance > 5:
                total_penalty += (distance - 5) ** 2 / 1000.0
            curr_color = curr["color"][:3]
            prev_color = prev["color"][:3]
            color_dist = sum((a - b) ** 2 for a, b in zip(curr_color, prev_color)) ** 0.5
            if color_dist > 0.1:
                total_penalty += (color_dist - 0.1) ** 2
            n_compared += 1
        return total_penalty / max(1, n_compared)

    def _get_center(self, shape):
        if "cx" in shape and "cy" in shape:
            return shape["cx"], shape["cy"]
        if "x" in shape and "y" in shape and "w" in shape and "h" in shape:
            return shape["x"] + shape["w"] / 2, shape["y"] + shape["h"] / 2
        if "verts" in shape and shape["verts"]:
            return (np.mean([v[0] for v in shape["verts"]]),
                    np.mean([v[1] for v in shape["verts"]]))
        return 0, 0

    def update_after_frame(self, individual, rendered):
        self.previous_individual = [ShapeFactory.deep_copy(s) for s in individual]
        self.previous_render = rendered.copy()
        self.frame_count += 1

    def apply_motion_blur(self, current_frame, previous_frame):
        if previous_frame is None or not self.motion_blur:
            return current_frame
        alpha = self.motion_blur_strength
        blended = (current_frame.astype(np.float32) * (1 - alpha) +
                   previous_frame.astype(np.float32) * alpha)
        return blended.astype(np.uint8)

    def is_keyframe(self):
        if self.keyframe_interval <= 1: return True
        return self.frame_count % self.keyframe_interval == 0


# ---------------------------------------------------------------------------
# Camera and animation helpers
# ---------------------------------------------------------------------------
class CameraAnimator:
    def __init__(self, canvas_size, mode="subtle"):
        self.canvas_size = canvas_size
        self.mode = mode
        self.total_frames = 0

    def set_total_frames(self, n):
        self.total_frames = n

    def apply_transform(self, frame, frame_idx):
        if self.total_frames == 0:
            return frame
        t = frame_idx / self.total_frames
        if self.mode == "subtle":
            scale = 1.0 + 0.05 * t
            tx = 10 * np.sin(2 * np.pi * t)
            ty = 5 * np.cos(2 * np.pi * t * 0.5)
            rot = 0.5 * np.sin(2 * np.pi * t)
        elif self.mode == "dynamic":
            scale = 1.0 + 0.15 * np.sin(2 * np.pi * t)
            tx = 30 * np.sin(4 * np.pi * t)
            ty = 20 * np.cos(3 * np.pi * t)
            rot = 2.0 * np.sin(2 * np.pi * t)
        elif self.mode == "cinematic":
            scale = 1.0 + 0.2 * t
            tx = 40 * t
            ty = -20 * t
            rot = 1.0 * t
        else:
            return frame
        img = Image.fromarray(frame)
        w, h = img.size
        if abs(rot) > 0.01:
            img = img.rotate(rot, resample=Image.BICUBIC, expand=False)
        center = (w / 2, h / 2)
        cos_r = np.cos(np.deg2rad(rot))
        sin_r = np.sin(np.deg2rad(rot))
        coeffs = (
            scale * cos_r,
            -scale * sin_r,
            scale * sin_r * center[1] - scale * cos_r * center[0] + center[0] + tx,
            scale * sin_r,
            scale * cos_r,
            -scale * cos_r * center[1] - scale * sin_r * center[0] + center[1] + ty,
        )
        img = img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
        return np.asarray(img)


class FrameInterpolator:
    @staticmethod
    def smooth_frames(frames, strength=0.3):
        if len(frames) < 3 or strength <= 0:
            return frames
        smoothed = [frames[0]]
        for i in range(1, len(frames) - 1):
            prev_frame = frames[i - 1].astype(np.float32)
            curr_frame = frames[i].astype(np.float32)
            next_frame = frames[i + 1].astype(np.float32)
            blended = (1 - strength) * curr_frame + strength * (prev_frame + next_frame) / 2
            smoothed.append(blended.astype(np.uint8))
        smoothed.append(frames[-1])
        return smoothed


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
class PostProcessor:
    @staticmethod
    def sharpen(frame, strength=0.5):
        if strength <= 0: return frame
        img = Image.fromarray(frame)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(strength * 150), threshold=2))
        return np.asarray(img)

    @staticmethod
    def color_grade(frame, preset="cinematic"):
        if preset == "none": return frame
        arr = frame.astype(np.float32)
        if preset == "cinematic":
            luminance = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
            shadow_mask = np.clip(1.0 - luminance / 128.0, 0, 1)
            highlight_mask = np.clip((luminance - 128.0) / 128.0, 0, 1)
            arr[:,:,0] -= shadow_mask * 8
            arr[:,:,1] += shadow_mask * 4
            arr[:,:,2] += shadow_mask * 12
            arr[:,:,0] += highlight_mask * 10
            arr[:,:,1] += highlight_mask * 4
            arr[:,:,2] -= highlight_mask * 8
        elif preset == "warm":
            arr[:,:,0] *= 1.05; arr[:,:,1] *= 1.02; arr[:,:,2] *= 0.95
        elif preset == "cool":
            arr[:,:,0] *= 0.95; arr[:,:,1] *= 1.02; arr[:,:,2] *= 1.08
        elif preset == "vintage":
            gray = np.mean(arr, axis=2, keepdims=True)
            arr = arr * 0.7 + gray * 0.3
            arr[:,:,0] *= 1.08; arr[:,:,2] *= 0.92
            arr = arr * 0.9 + 15
        elif preset == "noir":
            gray = np.mean(arr, axis=2, keepdims=True)
            gray = np.clip((gray - 128) * 1.3 + 128, 0, 255)
            arr = np.repeat(gray, 3, axis=2)
            arr[:,:,0] *= 1.02; arr[:,:,2] *= 0.98
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def vignette(frame, strength=0.4):
        if strength <= 0: return frame
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = dist / max_dist
        mask = 1.0 - strength * (dist ** 2)
        mask = np.clip(mask, 0, 1)
        result = frame.astype(np.float32)
        for c in range(3):
            result[:,:,c] *= mask
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def film_grain(frame, strength=0.05):
        if strength <= 0: return frame
        noise = np.random.normal(0, strength * 50, frame.shape).astype(np.float32)
        result = frame.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_pipeline(frame, settings):
        if settings.get("sharpen", 0) > 0:
            frame = PostProcessor.sharpen(frame, settings["sharpen"])
        if settings.get("color_grade", "none") != "none":
            frame = PostProcessor.color_grade(frame, settings["color_grade"])
        if settings.get("vignette", 0) > 0:
            frame = PostProcessor.vignette(frame, settings["vignette"])
        if settings.get("film_grain", 0) > 0:
            frame = PostProcessor.film_grain(frame, settings["film_grain"])
        return frame


class CaptionBurner:
    STYLES = {
        "minimal": {"font_size": 24, "color": (255, 255, 255), "bg_color": None, "position": "bottom", "padding": 20},
        "cinematic": {"font_size": 32, "color": (255, 255, 255), "bg_color": (0, 0, 0, 160), "position": "bottom", "padding": 30},
        "bold": {"font_size": 40, "color": (255, 255, 100), "bg_color": (0, 0, 0, 200), "position": "center", "padding": 40},
    }

    @staticmethod
    def burn(frame, caption, style="cinematic"):
        if not caption: return frame
        cfg = CaptionBurner.STYLES.get(style, CaptionBurner.STYLES["cinematic"])
        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", cfg["font_size"])
        except Exception:
            try: font = ImageFont.truetype("arial.ttf", cfg["font_size"])
            except Exception: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w, h = img.size
        pad = cfg["padding"]
        if cfg["position"] == "bottom":
            x = (w - tw) // 2; y = h - th - pad
        elif cfg["position"] == "top":
            x = (w - tw) // 2; y = pad
        else:
            x = (w - tw) // 2; y = (h - th) // 2
        if cfg["bg_color"]:
            bg_rect = [x - pad//2, y - pad//3, x + tw + pad//2, y + th + pad//3]
            draw.rectangle(bg_rect, fill=cfg["bg_color"])
        draw.text((x, y), caption, fill=cfg["color"], font=font)
        result = Image.alpha_composite(img, overlay).convert("RGB")
        return np.asarray(result)


# ---------------------------------------------------------------------------
# MP4 Encoder (web + hd only)
# ---------------------------------------------------------------------------
class MP4Encoder:
    QUALITY_WEB = "web"
    QUALITY_HD = "hd"

    TIER_CONFIGS = {
        "web": {
            "max_width": 1280, "crf": 23, "preset": "medium",
            "audio_bitrate": "128k", "pixel_format": "yuv420p",
            "profile": "main", "level": "3.1",
        },
        "hd": {
            "max_width": 1920, "crf": 18, "preset": "slow",
            "audio_bitrate": "192k", "pixel_format": "yuv420p",
            "profile": "high", "level": "4.0",
        },
    }

    @staticmethod
    def encode(frames, output_path, fps=24, audio_path=None, quality="hd",
               metadata=None, chapters=None):
        if not frames: return None
        config = MP4Encoder.TIER_CONFIGS.get(quality, MP4Encoder.TIER_CONFIGS["hd"])
        h, w = frames[0].shape[:2]
        if config["max_width"] and w > config["max_width"]:
            scale = config["max_width"] / w
            new_w = config["max_width"]
            new_h = int(h * scale)
            new_h = new_h - (new_h % 2)
            new_w = new_w - (new_w % 2)
            resized = []
            for f in frames:
                img = Image.fromarray(f).resize((new_w, new_h), Image.LANCZOS)
                resized.append(np.asarray(img))
            frames = resized
            h, w = new_h, new_w
        if w % 2 != 0 or h % 2 != 0:
            w = w - (w % 2); h = h - (h % 2)
            frames = [f[:h, :w] for f in frames]
        tmp_video = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
        cmd = ["ffmpeg", "-y", "-i", tmp_video]
        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", audio_path])
        cmd.extend([
            "-c:v", "libx264", "-preset", config["preset"],
            "-crf", str(config["crf"]),
            "-profile:v", config["profile"], "-level", config["level"],
            "-pix_fmt", config["pixel_format"],
            "-movflags", "+faststart",
        ])
        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-c:a", "aac", "-b:a", config["audio_bitrate"],
                       "-ar", "48000", "-ac", "2", "-shortest"])
        else:
            cmd.extend(["-an"])
        if metadata:
            for key, value in metadata.items():
                cmd.extend(["-metadata", f"{key}={value}"])
        chapter_file = None
        if chapters and len(chapters) > 1:
            chapter_file = output_path + ".chapters.txt"
            MP4Encoder._write_chapter_file(chapters, chapter_file, len(frames) / fps)
            cmd.extend(["-i", chapter_file, "-map_chapters", "1"])
        cmd.append(output_path)
        try:
            import subprocess
            subprocess.run(cmd, check=True, capture_output=True)
            try:
                os.remove(tmp_video)
                if chapter_file and os.path.exists(chapter_file):
                    os.remove(chapter_file)
            except OSError: pass
            return output_path
        except (FileNotFoundError, Exception) as e:
            try: os.replace(tmp_video, output_path)
            except OSError: pass
            return output_path

    @staticmethod
    def _write_chapter_file(chapters, path, total_duration):
        with open(path, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            for i, (time_sec, title) in enumerate(chapters):
                start_ms = int(time_sec * 1000)
                end_ms = int(chapters[i + 1][0] * 1000) if i + 1 < len(chapters) else int(total_duration * 1000)
                f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\ntitle={title}\n\n")

    @staticmethod
    def generate_thumbnail(frames, output_path, time_position=0.5):
        if not frames: return None
        idx = int(len(frames) * time_position)
        idx = max(0, min(idx, len(frames) - 1))
        img = Image.fromarray(frames[idx])
        img.thumbnail((320, 240), Image.LANCZOS)
        img.save(output_path, optimize=True)
        return output_path


# ---------------------------------------------------------------------------
# Music generator (with moods)
# ---------------------------------------------------------------------------
class MusicGenerator:
    MOODS = ["ambient", "melancholic", "hopeful", "mysterious", "epic"]

    @staticmethod
    def generate(duration_seconds, sample_rate=44100, path="music.wav",
                 mood="ambient", image_index=0):
        n = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, n, endpoint=False)
        signal = np.zeros(n, dtype=np.float32)
        seed_offset = image_index * 37
        if mood == "ambient":
            freqs = [110.0, 130.81, 164.81, 220.0]
            for i, f in enumerate(freqs):
                mod = 0.5 + 0.5 * np.sin(2 * np.pi * (f / 8.0) * t + seed_offset)
                signal += 0.12 * np.sin(2 * np.pi * f * t) * mod
            signal += 0.04 * np.sin(2 * np.pi * 440.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
        elif mood == "melancholic":
            freqs = [110.0, 130.81, 164.81]
            for f in freqs:
                env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
                signal += 0.15 * np.sin(2 * np.pi * f * t) * env
            signal += 0.08 * np.sin(2 * np.pi * 55.0 * t)
        elif mood == "hopeful":
            freqs = [130.81, 164.81, 196.0, 261.63]
            for f in freqs:
                mod = 0.6 + 0.4 * np.sin(2 * np.pi * (f / 6.0) * t)
                signal += 0.1 * np.sin(2 * np.pi * f * t) * mod
            signal += 0.03 * np.sin(2 * np.pi * 880.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
        elif mood == "mysterious":
            freqs = [110.0, 146.83, 185.0]
            for f in freqs:
                mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t + seed_offset)
                signal += 0.12 * np.sin(2 * np.pi * f * t) * mod
        elif mood == "epic":
            freqs = [82.41, 110.0, 164.81, 220.0, 329.63]
            for i, f in enumerate(freqs):
                amp = 0.15 / (i + 1)
                signal += amp * np.sin(2 * np.pi * f * t)
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
            signal *= (0.7 + 0.3 * pulse)
        fade = min(int(sample_rate * 2.0), n // 4)
        signal[:fade] *= np.linspace(0, 1, fade)
        signal[-fade:] *= np.linspace(1, 0, fade)
        peak = np.max(np.abs(signal))
        if peak > 0: signal = signal / peak * 0.75
        signal16 = (signal * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
            wf.writeframes(signal16.tobytes())
        return path


# ---------------------------------------------------------------------------
# Genetic Engine
# ---------------------------------------------------------------------------
class GeneticEngine:
    def __init__(self, target, canvas_size=(256,256), num_shapes=80, shape_mode="triangle",
                 n_sides=5, population_size=80, mutation_rate=0.02, elite_count=4,
                 alpha=128, use_smart_init=True, use_hill_climb=True, hill_climb_steps=20,
                 use_adaptive_mutation=True, use_ssim=False, initial_individual=None,
                 seed=None, color_space="rgb",
                 use_residual_guidance=True, use_multiscale_fitness=False,
                 use_annealing=True, use_de_blend=True, use_shape_split=True,
                 allow_dynamic_shape_count=False, use_fitness_cache=True,
                 use_diversity_injection=True, use_importance_tracking=True,
                 use_gradient_mutation=True,
                 temporal_coherence=0.3, temporal_engine=None,
                 evaluation_scale=1.0):
        self.target = target; self.canvas_size = canvas_size; self.num_shapes = num_shapes
        self.shape_mode = shape_mode; self.n_sides = n_sides; self.population_size = population_size
        self.mutation_rate = mutation_rate; self.elite_count = elite_count; self.alpha = alpha
        self.color_space = color_space
        self.use_smart_init = use_smart_init; self.use_hill_climb = use_hill_climb
        self.hill_climb_steps = hill_climb_steps; self.use_adaptive_mutation = use_adaptive_mutation
        self.use_ssim = use_ssim
        self.use_residual_guidance = use_residual_guidance
        self.use_multiscale_fitness = use_multiscale_fitness
        self.use_annealing = use_annealing; self.use_de_blend = use_de_blend
        self.use_shape_split = use_shape_split
        self.allow_dynamic_shape_count = allow_dynamic_shape_count
        self.use_fitness_cache = use_fitness_cache
        self.use_diversity_injection = use_diversity_injection
        self.use_importance_tracking = use_importance_tracking
        self.use_gradient_mutation = use_gradient_mutation
        self.temporal_coherence = temporal_coherence
        self.temporal_engine = temporal_engine
        self.evaluation_scale = evaluation_scale
        if seed is not None and seed > 0:
            random.seed(seed); np.random.seed(seed)
        t_img = Image.fromarray(target).convert("RGB").resize(canvas_size)
        self.target_small = np.asarray(t_img).astype(np.float32)/255.0
        self.target_flat = self.target_small.flatten()
        self.bg_color = tuple(float(c) for c in np.mean(self.target_small, axis=(0, 1)))
        self.edge_prob = None
        if use_smart_init: self._compute_edge_map()
        self.is_mixed = (shape_mode == "mixed")
        self.residual_map = ResidualMap(canvas_size) if use_residual_guidance else None
        self.multiscale = MultiScaleFitness(use_ssim=use_ssim) if use_multiscale_fitness else None
        self.annealing = AnnealingAcceptor() if use_annealing else None
        self.fitness_cache = FitnessCache() if use_fitness_cache else None
        self.diversity_monitor = DiversityMonitor() if use_diversity_injection else None
        self.importance_tracker = ShapeImportanceTracker() if use_importance_tracking else None
        if initial_individual is not None:
            self.population = [initial_individual]
            for _ in range(population_size-1):
                self.population.append(self._mutate_individual(self._copy_individual(initial_individual)))
        elif use_smart_init:
            self.population = [self._smart_individual() for _ in range(population_size)]
        else:
            self.population = [self._random_individual() for _ in range(population_size)]
        self.best_fitness = float("inf"); self.best_individual = None; self.generation = 0
        self.current_mutation_rate = mutation_rate
        self.fitness_history = []; self.stagnation = 0

    def _compute_edge_map(self):
        gray = np.mean(self.target_small, axis=2)
        gx, gy = np.gradient(gray, axis=1), np.gradient(gray, axis=0)
        mag = np.sqrt(gx*gx+gy*gy)+0.01; self.edge_prob = mag/mag.sum()

    def _pick_kind(self):
        return ShapeFactory.random_kind() if self.is_mixed else self.shape_mode

    def _smart_shape_placement(self):
        kind = self._pick_kind()
        w, h = self.canvas_size
        if self.residual_map is not None and self.residual_map.placement_prob is not None:
            cx, cy = self.residual_map.sample_position()
            iy = int(np.clip(cy, 0, self.target_small.shape[0] - 1))
            ix = int(np.clip(cx, 0, self.target_small.shape[1] - 1))
            if self.color_space == "hsv":
                color = ColorSpace.sample_color_hsv(self.target_small, iy, ix) + (self.alpha / 255.0,)
            else:
                color = tuple(self.target_small[iy, ix]) + (self.alpha / 255.0,)
            shape = ShapeFactory.random_shape(kind, self.canvas_size, alpha=self.alpha,
                                              n_sides=self.n_sides, color_space=self.color_space)
            if "verts" in shape:
                old_cx = np.mean([v[0] for v in shape["verts"]])
                old_cy = np.mean([v[1] for v in shape["verts"]])
                dx = cx - old_cx; dy = cy - old_cy
                shape["verts"] = [(np.clip(v[0]+dx, 0, w-1), np.clip(v[1]+dy, 0, h-1))
                                   for v in shape["verts"]]
            else:
                for k in ("cx", "cy"):
                    if k in shape: shape[k] = cx if k == "cx" else cy
            shape["color"] = color
            return shape
        else:
            return ShapeFactory.random_shape(
                kind, self.canvas_size,
                target_small=self.target_small if self.use_smart_init else None,
                edge_prob=self.edge_prob if self.use_smart_init else None,
                alpha=self.alpha, n_sides=self.n_sides, color_space=self.color_space)

    def _random_shape(self):
        return ShapeFactory.random_shape(self._pick_kind(), self.canvas_size,
            target_small=self.target_small if self.use_smart_init else None,
            edge_prob=self.edge_prob if self.use_smart_init else None,
            alpha=self.alpha, n_sides=self.n_sides, color_space=self.color_space)

    def _smart_individual(self): return [self._smart_shape_placement() for _ in range(self.num_shapes)]
    def _random_individual(self): return [ShapeFactory.random_shape(self._pick_kind(), self.canvas_size, alpha=self.alpha, n_sides=self.n_sides, color_space=self.color_space) for _ in range(self.num_shapes)]
    def _copy_individual(self, ind): return [ShapeFactory.deep_copy(s) for s in ind]

    def _render_individual(self, ind):
        w, h = self.canvas_size
        bg = tuple(int(c * 255) for c in self.bg_color)
        img = Image.new("RGBA", (w, h), bg + (255,))
        draw = ImageDraw.Draw(img, "RGBA")
        for s in ind: ShapeFactory.render(draw, s, color_space=self.color_space)
        return np.asarray(img.convert("RGB")).astype(np.float32)/255.0

    def _ssim(self, img1, img2):
        C1, C2 = 0.01**2, 0.03**2
        def blur(a): return np.array(Image.fromarray((a*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)/255
        m1, m2 = blur(img1), blur(img2)
        s1 = np.maximum(blur(img1*img1)-m1*m1,0); s2 = np.maximum(blur(img2*img2)-m2*m2,0)
        s12 = blur(img1*img2)-m1*m2
        return float(np.mean((2*m1*m2+C1)*(2*s12+C2)/((m1*m1+m2*m2+C1)*(s1+s2+C2))))

    def _fitness(self, ind):
        if self.fitness_cache is not None:
            cached = self.fitness_cache.get(ind)
            if cached is not None: return cached
        rendered = self._render_individual(ind)
        if self.use_multiscale_fitness and self.multiscale is not None:
            accuracy = self.multiscale.evaluate(rendered, self.target_small)
        elif self.use_ssim:
            accuracy = 1.0 - self._ssim(rendered, self.target_small)
        else:
            accuracy = float(np.mean((rendered.flatten() - self.target_flat) ** 2))
        temporal_penalty = 0.0
        if (self.temporal_engine is not None and
            self.temporal_engine.previous_individual is not None and
            self.temporal_coherence > 0):
            temporal_penalty = (self.temporal_coherence *
                                self.temporal_engine.temporal_coherence_penalty(
                                    ind, self.temporal_engine.previous_individual))
        fitness = accuracy + temporal_penalty
        if self.fitness_cache is not None:
            self.fitness_cache.put(ind, fitness)
        return fitness

    def _update_adaptive_mutation(self, cb):
        self.fitness_history.append(cb)
        if len(self.fitness_history)>20: self.fitness_history.pop(0)
        if len(self.fitness_history)>=2:
            if self.fitness_history[-1]<self.fitness_history[-2]-1e-6:
                self.current_mutation_rate *= 0.95; self.stagnation = 0
            else:
                self.stagnation += 1
                if self.stagnation>5: self.current_mutation_rate *= 1.1; self.stagnation = 0
        self.current_mutation_rate = np.clip(self.current_mutation_rate, 0.001, 0.3)

    def _hill_climb(self, ind, cf):
        bf, bi = cf, self._copy_individual(ind)
        if (hasattr(self, 'importance_tracker') and
            self.importance_tracker is not None and
            self.importance_tracker.importance_scores):
            ranked = sorted(self.importance_tracker.importance_scores.items(),
                           key=lambda x: x[1], reverse=True)
            priority_indices = [idx for idx, _ in ranked[:5]]
        else:
            priority_indices = list(range(len(bi)))
            random.shuffle(priority_indices)
        for step in range(self.hill_climb_steps):
            candidate = self._copy_individual(bi)
            if step < len(priority_indices):
                idx = priority_indices[step]
            else:
                idx = random.randrange(len(candidate))
            candidate[idx] = ShapeFactory.micro_mutate(candidate[idx], self.canvas_size, self.color_space)
            f = self._fitness(candidate)
            if f < bf: bf, bi = f, candidate
        return bf, bi

    def _shape_center(self, shape):
        if "cx" in shape and "cy" in shape: return shape["cx"], shape["cy"]
        if "x" in shape and "y" in shape and "w" in shape and "h" in shape:
            return shape["x"] + shape["w"] / 2, shape["y"] + shape["h"] / 2
        if "verts" in shape and shape["verts"]:
            return np.mean([v[0] for v in shape["verts"]]), np.mean([v[1] for v in shape["verts"]])
        return self.canvas_size[0] / 2, self.canvas_size[1] / 2

    def _shrink_shape(self, shape, factor):
        cx, cy = self._shape_center(shape)
        if "verts" in shape:
            shape["verts"] = [(cx + (v[0] - cx) * factor, cy + (v[1] - cy) * factor) for v in shape["verts"]]
        for k in ("rx", "ry", "r", "w", "h", "outer_r", "inner_r", "size", "thickness"):
            if k in shape: shape[k] *= factor

    def _offset_shape(self, shape, dx=0, dy=0):
        if "verts" in shape:
            shape["verts"] = [(v[0] + dx, v[1] + dy) for v in shape["verts"]]
        for k in ("cx", "cy", "x", "y"):
            if k in shape:
                if k in ("cx", "x"): shape[k] += dx
                else: shape[k] += dy

    def _mutate_individual(self, ind):
        ind = self._copy_individual(ind)
        if self.use_shape_split and random.random() < 0.05 and len(ind) < self.num_shapes * 1.5:
            if self.residual_map is not None:
                worst_idx = -1; worst_error = -1
                for i, shape in enumerate(ind):
                    cx, cy = self._shape_center(shape)
                    err = self.residual_map.get_error_at(cx, cy)
                    if err > worst_error: worst_error = err; worst_idx = i
                if worst_idx >= 0 and worst_error > 0.1:
                    original = ind[worst_idx]
                    clone = ShapeFactory.deep_copy(original)
                    self._shrink_shape(original, 0.7); self._shrink_shape(clone, 0.7)
                    self._offset_shape(clone, dx=random.uniform(-10, 10), dy=random.uniform(-10, 10))
                    ind[worst_idx] = original; ind.append(clone)
                    return ind
        if self.allow_dynamic_shape_count and len(ind) > self.num_shapes // 2:
            if random.random() < 0.03 and self.residual_map is not None:
                best_idx = -1; best_error = float("inf")
                for i, shape in enumerate(ind):
                    cx, cy = self._shape_center(shape)
                    err = self.residual_map.get_error_at(cx, cy)
                    if err < best_error: best_error = err; best_idx = i
                if best_idx >= 0:
                    ind.pop(best_idx); return ind
        idx = random.randrange(len(ind))
        if self.importance_tracker is not None:
            importance = self.importance_tracker.get_importance(idx)
            importance_factor = 0.3 + 1.4 * (1.0 - importance)
        else:
            importance_factor = 1.0
        if self.residual_map is not None:
            cx, cy = self._shape_center(ind[idx])
            local_error = self.residual_map.get_error_at(cx, cy)
            local_mutation = self.current_mutation_rate * (0.5 + 2.0 * local_error) * importance_factor
        else:
            local_mutation = self.current_mutation_rate * importance_factor
        if random.random() < 0.05 and len(ind) > 1:
            idx2 = random.randrange(len(ind))
            if idx2 != idx:
                ind[idx], ind[idx2] = ind[idx2], ind[idx]
                return ind
        if (self.use_gradient_mutation and self.residual_map is not None
                and random.random() < 0.3):
            # Simple gradient-directed: move toward higher error
            cx, cy = self._shape_center(ind[idx])
            ix = int(np.clip(cx, 1, self.canvas_size[0] - 2))
            iy = int(np.clip(cy, 1, self.canvas_size[1] - 2))
            em = self.residual_map.error_map
            if em is not None:
                gx = float(em[iy, min(ix+1, self.canvas_size[0]-1)] - em[iy, max(ix-1, 0)])
                gy = float(em[min(iy+1, self.canvas_size[1]-1), ix] - em[max(iy-1, 0), ix])
                step = local_mutation * 100
                dx = gx * step; dy = gy * step
                s = ShapeFactory.deep_copy(ind[idx])
                if "cx" in s:
                    s["cx"] = np.clip(s["cx"] + dx, 0, self.canvas_size[0])
                    s["cy"] = np.clip(s["cy"] + dy, 0, self.canvas_size[1])
                elif "x" in s:
                    s["x"] = np.clip(s["x"] + dx, 0, self.canvas_size[0])
                    s["y"] = np.clip(s["y"] + dy, 0, self.canvas_size[1])
                elif "verts" in s:
                    s["verts"] = [(np.clip(vx + dx, 0, self.canvas_size[0]-1),
                                   np.clip(vy + dy, 0, self.canvas_size[1]-1)) for vx, vy in s["verts"]]
                ind[idx] = s
                return ind
        ind[idx] = ShapeFactory.mutate(ind[idx], self.canvas_size, local_mutation,
            self.n_sides, self.is_mixed, self.color_space)
        return ind

    def _crossover(self, parent_a, parent_b):
        if self.use_de_blend and len(parent_a) == len(parent_b):
            target = parent_a if random.random() < 0.5 else parent_b
            other_a = parent_b if target is parent_a else parent_a
            other_b = self._copy_individual(other_a)
            if other_b:
                idx = random.randrange(len(other_b))
                other_b[idx] = ShapeFactory.mutate(other_b[idx], self.canvas_size,
                                                    0.1, self.n_sides, False, self.color_space)
            child = []
            for pa, pb, pt in zip(parent_a, parent_b, target):
                if random.random() < 0.7:
                    child.append(DEBlender.blend_shapes(pa, pb, pt, F=0.5, CR=0.7))
                else:
                    child.append(ShapeFactory.deep_copy(pa if random.random() < 0.5 else pb))
            return child
        else:
            return [ShapeFactory.deep_copy(pa if random.random() < 0.5 else pb)
                    for pa, pb in zip(parent_a, parent_b)]

    def _tournament(self, scored, k=None):
        if k is None:
            phase = min(1.0, self.generation / 500.0)
            k = int(2 + 6 * phase)
        c = random.sample(scored, min(k, len(scored))); c.sort(key=lambda x: x[0])
        return c[0][1]

    def upscale_individual(self, ind, factor): return [ShapeFactory.upscale(s, factor) for s in ind]

    def step(self):
        scored = [(self._fitness(i), i) for i in self.population]
        scored.sort(key=lambda x: x[0])
        bf, bi = scored[0]
        if self.use_adaptive_mutation: self._update_adaptive_mutation(bf)
        if self.use_hill_climb: bf, bi = self._hill_climb(bi, bf)
        if self.use_annealing and self.annealing is not None:
            if self.annealing.accept(self.best_fitness, bf):
                if bf < self.best_fitness:
                    self.best_fitness = bf; self.best_individual = bi
            self.annealing.cool()
        else:
            if bf < self.best_fitness:
                self.best_fitness = bf; self.best_individual = bi
        new_pop = [self._copy_individual(self.best_individual)]
        for i in range(1, min(self.elite_count, len(scored))):
            new_pop.append(scored[i][1])
        while len(new_pop) < self.population_size:
            pa = self._tournament(scored); pb = self._tournament(scored)
            child = self._mutate_individual(self._crossover(pa, pb))
            new_pop.append(child)
        if self.diversity_monitor is not None:
            div = self.diversity_monitor.compute_diversity(new_pop)
            if self.diversity_monitor.should_inject(self.generation):
                n_inject = self.diversity_monitor.inject_count
                for _ in range(min(n_inject, len(new_pop) - self.elite_count)):
                    new_pop.pop()
                    new_pop.append(self._smart_individual() if self.use_smart_init
                                   else self._random_individual())
        self.population = new_pop
        self.generation += 1
        if self.use_residual_guidance and self.residual_map is not None:
            best_rendered = self._render_individual(self.best_individual)
            self.residual_map.update(best_rendered, self.target_small)
        if (self.use_importance_tracking and self.importance_tracker is not None
                and self.generation % 50 == 0):
            self.importance_tracker.compute_importance(self.best_individual, self._fitness, n_samples=15)
        return self.generation, self.best_fitness, (self._render_individual(self.best_individual)*255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Batch worker
# ---------------------------------------------------------------------------
class BatchWorker(QObject):
    frame_ready = Signal(int, int, float, np.ndarray)
    image_started = Signal(int, str)
    image_finished = Signal(int, str, float, int, str)
    image_failed = Signal(int, str)
    all_done = Signal(str, int, int, str)
    error = Signal(str)
    log = Signal(str)
    eta_update = Signal(str)
    progress_update = Signal(int)
    image_progress = Signal(int, int, int, float)
    fitness_point = Signal(int, int, float)

    def __init__(self, entries, params, output_dir, name_template="{name}_{shape}",
                 preset_name="custom"):
        super().__init__()
        self.entries = entries; self.params = params
        self.output_dir = output_dir; self.name_template = name_template
        self.preset_name = preset_name
        self.timing = TimingController(params)
        self._stop = False; self._paused = False

    def request_stop(self): self._stop = True
    def request_pause(self): self._paused = True
    def request_resume(self): self._paused = False

    def run(self):
        try: self._do_run()
        except Exception as e:
            import traceback; self.error.emit(f"Error: {e}\n{traceback.format_exc()}")

    def _make_engine(self, target_arr, canvas_size, initial_individual=None, temporal_engine=None):
        seed = int(self.params.get("seed", 0))
        return GeneticEngine(
            target=target_arr, canvas_size=canvas_size,
            num_shapes=int(self.params["num_shapes"]), shape_mode=self.params["shape_mode"],
            n_sides=int(self.params["n_sides"]), population_size=int(self.params["population_size"]),
            mutation_rate=float(self.params["mutation_rate"]), elite_count=int(self.params["elite_count"]),
            alpha=int(self.params["alpha"]),
            use_smart_init=self.params.get("use_smart_init", True),
            use_hill_climb=self.params.get("use_hill_climb", True),
            hill_climb_steps=int(self.params.get("hill_climb_steps", 20)),
            use_adaptive_mutation=self.params.get("use_adaptive_mutation", True),
            use_ssim=self.params.get("use_ssim", False),
            initial_individual=initial_individual,
            color_space=self.params.get("color_space", "rgb"),
            use_residual_guidance=self.params.get("use_residual_guidance", True),
            use_multiscale_fitness=self.params.get("use_multiscale_fitness", False),
            use_annealing=self.params.get("use_annealing", True),
            use_de_blend=self.params.get("use_de_blend", True),
            use_shape_split=self.params.get("use_shape_split", True),
            allow_dynamic_shape_count=self.params.get("allow_dynamic_shape_count", False),
            use_fitness_cache=self.params.get("use_fitness_cache", True),
            use_diversity_injection=self.params.get("use_diversity_injection", True),
            use_importance_tracking=self.params.get("use_importance_tracking", True),
            use_gradient_mutation=self.params.get("use_gradient_mutation", True),
            temporal_coherence=float(self.params.get("temporal_coherence", 0.3)),
            temporal_engine=temporal_engine,
            evaluation_scale=float(self.params.get("evaluation_scale", 1.0)),
            seed=seed if seed > 0 else None)

    def _do_run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        canvas = (int(self.params["canvas_w"]), int(self.params["canvas_h"]))
        fps = self.timing.fps
        display_size = (int(self.params["display_w"]), int(self.params["display_h"]))
        use_multiscale = self.params.get("use_multiscale", False)
        multiscale_ratio = self.params.get("multiscale_ratio", 0.3)
        export_png = self.params.get("export_png", False)
        quality = self.params.get("quality_tier", "hd")
        post_settings = {
            "sharpen": float(self.params.get("post_sharpen", 0.3)),
            "color_grade": self.params.get("post_color_grade", "none"),
            "vignette": float(self.params.get("post_vignette", 0.2)),
            "film_grain": float(self.params.get("post_film_grain", 0.0)),
        }
        caption_style = self.params.get("caption_style", "none")
        generate_per_image = self.params.get("generate_per_image_mp4", True)
        generate_thumbnail = self.params.get("generate_thumbnail", True)
        audio_mood = self.params.get("audio_mood", "ambient")
        camera_mode = self.params.get("camera_mode", "cinematic")
        temporal_smooth = self.params.get("temporal_smooth", False)
        smooth_strength = float(self.params.get("smooth_strength", 0.3))
        total_images = len(self.entries)
        start_time = time.time()
        succeeded, failed = 0, 0
        stats = {"images": [], "total_time": 0, "avg_mse": 0, "best_mse": float("inf"),
                 "worst_mse": 0, "total_gens": 0}
        all_video_frames = []
        chapters = []
        current_time = 0.0

        if self.timing.intro_enabled and self.timing.intro_text:
            self.log.emit("🎬 Generating intro...")
            intro_frames = CardGenerator.generate(
                self.timing.intro_text, display_size[0]*2, display_size[1]+40,
                self.timing.intro_seconds, fps)
            all_video_frames.extend(intro_frames)
            current_time += self.timing.intro_seconds

        for i, entry in enumerate(self.entries):
            if self._stop: break
            entry.status = "running"; entry.started_at = time.time()
            entry.total_gens = self.timing.max_gens
            self.image_started.emit(i, entry.caption)
            self.log.emit(f"[{i+1}/{total_images}] Loading: {entry.url[:80]}")
            try:
                img = ImageDownloader.download(entry.url, timeout=15, max_retries=3)
                target_arr = np.asarray(img)
                entry.target_array = target_arr; entry.loaded = True
            except Exception as e:
                self.log.emit(f"  ✗ Failed: {e}")
                entry.status = "failed"; failed += 1
                self.image_failed.emit(i, str(e)); continue
            out_name = OutputNamer.generate(self.name_template, entry, i, self.params, self.preset_name)
            out_path = os.path.join(self.output_dir, out_name + ".mp4")
            entry.output_path = out_path
            self.log.emit(f"  → target MSE: {self.timing.target_mse}, max {self.timing.max_gens} gens")
            try:
                temporal = TemporalEngine(self.params)
                raw_frames = []
                fitness_history = []
                gens_run = 0
                eng = self._make_engine(target_arr, canvas, temporal_engine=temporal)
                for g in range(self.timing.max_gens):
                    if self._stop: break
                    while self._paused and not self._stop: QThread.msleep(100)
                    if self._stop: break
                    is_keyframe = temporal.is_keyframe()
                    if is_keyframe:
                        if g == 0 and temporal.previous_individual is not None:
                            seeded = temporal.create_seeded_population(eng, eng.population_size)
                            if seeded is not None: eng.population = seeded
                        gen, mse, fs = eng.step()
                    else:
                        if temporal.previous_individual is not None:
                            current_ind = [ShapeFactory.deep_copy(s) for s in temporal.previous_individual]
                            for _ in range(max(1, len(current_ind) // 20)):
                                idx = random.randrange(len(current_ind))
                                current_ind[idx] = ShapeFactory.micro_mutate(
                                    current_ind[idx], canvas, eng.color_space)
                            fs = (eng._render_individual(current_ind) * 255).astype(np.uint8)
                            mse = eng._fitness(current_ind); gen = g
                        else:
                            gen, mse, fs = eng.step()
                    gens_run += 1; fitness_history.append(mse)
                    entry.current_gen = gen; entry.current_mse = mse
                    self.image_progress.emit(i, gen, self.timing.max_gens, mse)
                    self.fitness_point.emit(i, gen, mse)
                    fi = Image.fromarray(fs).resize(display_size)
                    comp = self._make_composite(target_arr, np.asarray(fi), display_size,
                        f"Gen {gen} MSE {mse:.5f}", entry.caption)
                    if temporal.motion_blur and len(raw_frames) > 0:
                        comp = temporal.apply_motion_blur(comp, raw_frames[-1])
                    raw_frames.append(comp)
                    self.frame_ready.emit(i, gen, mse, comp)
                    self._emit_eta(i, total_images, gens_run, self.timing.max_gens, start_time)
                    temporal.update_after_frame(eng.best_individual,
                                                eng._render_individual(eng.best_individual))
                    if self.timing.should_stop_ga(gen, mse, fitness_history):
                        self.log.emit(f"  ⏹ Stop at gen {gen} (MSE: {mse:.6f})")
                        break
                    if g % 2 == 0: QThread.msleep(1)
                if self._stop or not raw_frames:
                    entry.status = "failed"; failed += 1
                    self.image_failed.emit(i, "Aborted"); continue
                image_frames = self.timing.resample_frames_to_duration(raw_frames)
                # Apply camera transforms
                if camera_mode != "none":
                    camera = CameraAnimator(display_size, mode=camera_mode)
                    camera.set_total_frames(len(image_frames))
                    image_frames = [camera.apply_transform(f, idx) for idx, f in enumerate(image_frames)]
                # Temporal smoothing
                if temporal_smooth:
                    image_frames = FrameInterpolator.smooth_frames(image_frames, strength=smooth_strength)
                # Post-processing
                image_frames = [PostProcessor.apply_pipeline(f, post_settings) for f in image_frames]
                # Caption burn-in
                if caption_style != "none" and entry.caption:
                    image_frames = [CaptionBurner.burn(f, entry.caption, caption_style) for f in image_frames]
                png_path = ""
                if export_png and image_frames:
                    png_path = os.path.join(self.output_dir, out_name + ".png")
                    Image.fromarray(image_frames[-1]).save(png_path)
                    entry.png_path = png_path
                hold_frames = self.timing.get_hold_frames()
                if hold_frames > 0 and image_frames:
                    final_frame = image_frames[-1]
                    for _ in range(hold_frames): image_frames.append(final_frame.copy())
                # Per-image MP4
                if generate_per_image:
                    per_image_audio = os.path.join(self.output_dir, out_name + ".tmp.wav")
                    MusicGenerator.generate(len(image_frames) / fps, path=per_image_audio,
                                            mood=audio_mood, image_index=i)
                    MP4Encoder.encode(image_frames, out_path, fps=fps,
                                     audio_path=per_image_audio, quality=quality,
                                     metadata={"title": entry.caption or out_name,
                                              "artist": "Genetic Evolution Studio",
                                              "comment": f"MSE: {entry.final_mse:.5f}"})
                    try: os.remove(per_image_audio)
                    except OSError: pass
                    if generate_thumbnail:
                        MP4Encoder.generate_thumbnail(image_frames,
                            os.path.join(self.output_dir, out_name + "_thumb.jpg"), 0.7)
                    self.log.emit(f"  🎬 Per-image MP4 → {out_name}.mp4")
                # Transitions for combined
                if all_video_frames and self.timing.transition_type != "cut":
                    trans_frames = self.timing.get_transition_frames()
                    if trans_frames > 0:
                        prev_final = all_video_frames[-1]
                        first_frame = image_frames[0]
                        for t_idx in range(trans_frames):
                            t = t_idx / max(1, trans_frames - 1)
                            if self.timing.transition_type == "crossfade":
                                tf = TransitionRenderer.crossfade(prev_final, first_frame, t)
                            elif self.timing.transition_type == "fade_black":
                                tf = (TransitionRenderer.fade_black(prev_final, t) if t < 0.5
                                      else TransitionRenderer.fade_black(first_frame, 1 - t))
                            elif self.timing.transition_type == "wipe":
                                tf = TransitionRenderer.wipe(prev_final, first_frame, t)
                            else: tf = first_frame
                            all_video_frames.append(tf)
                all_video_frames.extend(image_frames)
                chapters.append((current_time, entry.caption or f"Image {i+1}"))
                current_time += len(image_frames) / fps
                entry.status = "done"; entry.finished_at = time.time()
                entry.final_mse = entry.current_mse; entry.gens_run = gens_run
                succeeded += 1
                stats["images"].append({
                    "idx": i, "caption": entry.caption, "url": entry.url,
                    "mse": entry.final_mse, "gens": gens_run,
                    "mp4": out_path, "png": png_path,
                    "time": entry.finished_at - entry.started_at
                })
                stats["total_gens"] += gens_run
                stats["best_mse"] = min(stats["best_mse"], entry.final_mse)
                stats["worst_mse"] = max(stats["worst_mse"], entry.final_mse)
                self.image_finished.emit(i, out_path, entry.final_mse, gens_run, png_path)
                self.log.emit(f"  ✓ Done: {out_name} (MSE {entry.final_mse:.5f}, {gens_run} gens)")
                self.progress_update.emit(int(100*(i+1)/total_images))
            except Exception as e:
                import traceback
                self.log.emit(f"  ✗ Error: {e}\n{traceback.format_exc()}")
                entry.status = "failed"; failed += 1
                self.image_failed.emit(i, str(e))

        if self.timing.outro_enabled and self.timing.outro_text and not self._stop:
            self.log.emit("🎬 Generating outro...")
            outro_frames = CardGenerator.generate(
                self.timing.outro_text, display_size[0]*2, display_size[1]+40,
                self.timing.outro_seconds, fps)
            all_video_frames.extend(outro_frames)

        if all_video_frames:
            combined_path = os.path.join(self.output_dir, "combined_output.mp4")
            combined_duration = len(all_video_frames) / fps
            combined_audio = os.path.join(self.output_dir, "combined.tmp.wav")
            MusicGenerator.generate(combined_duration, path=combined_audio,
                                    mood=audio_mood, image_index=0)
            MP4Encoder.encode(all_video_frames, combined_path, fps=fps,
                             audio_path=combined_audio, quality=quality,
                             chapters=chapters,
                             metadata={"title": f"Genetic Evolution — {len(self.entries)} images",
                                      "artist": "Genetic Evolution Studio"})
            try: os.remove(combined_audio)
            except OSError: pass
            if generate_thumbnail:
                MP4Encoder.generate_thumbnail(all_video_frames,
                    os.path.join(self.output_dir, "combined_thumb.jpg"), 0.1)
            self.log.emit(f"🎬 Combined MP4 with {len(chapters)} chapters → combined_output.mp4")

        stats["total_time"] = time.time() - start_time
        stats["avg_mse"] = (stats["best_mse"] + stats["worst_mse"]) / 2 if succeeded > 0 else 0
        stats_path = os.path.join(self.output_dir, "batch_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        self.log.emit(f"📊 Stats saved → {stats_path}")
        self.all_done.emit(self.output_dir, succeeded, failed, stats_path)
        self._log_stats(stats, succeeded, failed)

    def _log_stats(self, stats, succeeded, failed):
        self.log.emit("\n" + "="*60)
        self.log.emit("📊 BATCH STATISTICS")
        self.log.emit("="*60)
        self.log.emit(f"  Total images:    {succeeded + failed}")
        self.log.emit(f"  Succeeded:       {succeeded}")
        self.log.emit(f"  Failed:          {failed}")
        self.log.emit(f"  Total time:      {stats['total_time']:.1f}s")
        self.log.emit(f"  Total gens:      {stats['total_gens']}")
        if succeeded > 0:
            self.log.emit(f"  Best MSE:        {stats['best_mse']:.6f}")
            self.log.emit(f"  Worst MSE:       {stats['worst_mse']:.6f}")
            self.log.emit(f"  Avg MSE:         {stats['avg_mse']:.6f}")
        self.log.emit("="*60 + "\n")

    def _emit_eta(self, img_idx, total_imgs, current_gen, total_gens, start_time):
        elapsed = time.time() - start_time
        total_gens_all = total_imgs * total_gens
        done_gens = img_idx * total_gens + current_gen
        if done_gens > 0:
            rate = elapsed / done_gens
            remaining = (total_gens_all - done_gens) * rate
            mins, secs = divmod(int(remaining), 60)
            hrs, mins = divmod(mins, 60)
            eta = f"{hrs}h {mins}m {secs}s" if hrs > 0 else f"{mins}m {secs}s"
        else:
            eta = "calculating..."
        overall_pct = int(100 * done_gens / max(1, total_gens_all))
        self.eta_update.emit(f"ETA: {eta}  |  {overall_pct}%")

    def _make_composite(self, target, evolved, display_size, status_text, caption):
        w, h = display_size
        t_img = Image.fromarray(target).resize((w, h))
        top = np.concatenate([np.asarray(t_img), evolved], axis=1)
        bar = np.zeros((40, top.shape[1], 3), dtype=np.uint8)
        pil = Image.fromarray(np.concatenate([top, bar], axis=0))
        draw = ImageDraw.Draw(pil)
        try: font = ImageFont.load_default()
        except Exception: font = None
        draw.text((8, h+8), status_text, fill=(255,255,255), font=font)
        if caption:
            cap = caption[:77]+"..." if len(caption)>80 else caption
            draw.text((w+8, h+8), cap, fill=(220,220,220), font=font)
        return np.asarray(pil)


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------
class ThumbnailListWidget(QListWidget):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(64, 48))
        self.setSpacing(2)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls()]
            if paths:
                self.files_dropped.emit(paths)
                event.acceptProposedAction(); return
        super().dropEvent(event)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genetic Image Evolution Studio v14.1")
        self.resize(1500, 950)
        self.entries: list[ImageEntry] = []
        self.worker = None
        self.worker_thread = None
        self._current_frame = None
        self.settings = QSettings("GeneticEvolution", "Studio")
        self._build_menu()
        self._build_ui()
        self._build_statusbar()
        self._setup_shortcuts()
        self._load_settings()
        self._load_sample_list()

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        for t, s, sl in [("&Load JSON…","Ctrl+J",self._on_load_json),
                          ("Load &Folder…","Ctrl+Shift+J",self._on_load_json_folder),
                          ("Load I&mages Folder…","Ctrl+M",self._on_load_images_folder),
                          ("Load &CSV…","Ctrl+Shift+C",self._on_load_csv),
                          ("&Expand URL Pattern…","Ctrl+P",self._on_expand_pattern)]:
            a = QAction(t,self); a.setShortcut(s); a.triggered.connect(sl); fm.addAction(a)
        fm.addSeparator()
        for t, sl in [("&Export Queue…",self._on_export_json),
                       ("&Save Job…",self._on_save_job),
                       ("&Load Job…",self._on_load_job)]:
            a = QAction(t,self); a.triggered.connect(sl); fm.addAction(a)
        fm.addSeparator()
        a = QAction("&Quit","Ctrl+Q",self); a.triggered.connect(self.close); fm.addAction(a)
        pm = mb.addMenu("&Presets")
        for name in BUILTIN_PRESETS:
            a = QAction(f"Load: {name}", self)
            a.triggered.connect(lambda checked, n=name: self._load_preset(n))
            pm.addAction(a)
        pm.addSeparator()
        a = QAction("Save Current as Preset…", self); a.triggered.connect(self._save_preset); pm.addAction(a)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal); root.addWidget(splitter)
        left = QWidget(); ll = QVBoxLayout(left)
        lg = QGroupBox("Image Queue"); lgl = QVBoxLayout(lg)
        self.list_widget = ThumbnailListWidget()
        self.list_widget.files_dropped.connect(self._on_files_dropped)
        self.list_widget.currentRowChanged.connect(self._on_queue_selection_changed)
        lgl.addWidget(self.list_widget)
        ar = QHBoxLayout()
        self.url_edit = QLineEdit(); self.url_edit.setPlaceholderText("Image URL or pattern")
        self.caption_edit = QLineEdit(); self.caption_edit.setPlaceholderText("Caption")
        ba = QPushButton("Add"); ba.clicked.connect(self._on_add_image)
        br = QPushButton("Remove"); br.clicked.connect(self._on_remove_image)
        ar.addWidget(self.url_edit,3); ar.addWidget(self.caption_edit,2); ar.addWidget(ba); ar.addWidget(br)
        lgl.addLayout(ar)
        jr = QHBoxLayout()
        for t, sl in [("📂 JSON",self._on_load_json),("📁 JSONs",self._on_load_json_folder),
                       ("🖼️ Images",self._on_load_images_folder),("📊 CSV",self._on_load_csv),
                       ("🔗 Pattern",self._on_expand_pattern)]:
            b = QPushButton(t); b.clicked.connect(sl); jr.addWidget(b)
        lgl.addLayout(jr)
        jr2 = QHBoxLayout()
        for t, sl in [("🗑 Clear",self._on_clear_queue),("💾 Export",self._on_export_json),
                       ("💼 Save Job",self._on_save_job),("📂 Load Job",self._on_load_job)]:
            b = QPushButton(t); b.clicked.connect(sl); jr2.addWidget(b)
        jr2.addStretch()
        self.count_label = QLabel("0 images"); self.count_label.setStyleSheet("color:#888;font-size:11px;")
        jr2.addWidget(self.count_label)
        lgl.addLayout(jr2)
        ll.addWidget(lg, 2)
        tabs = QTabWidget()
        # Tab 1: Shapes
        t1 = QWidget(); t1l = QFormLayout(t1)
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(SHAPE_KINDS + ["mixed"])
        self.combo_shape.setCurrentText("triangle")
        t1l.addRow("Shape mode:", self.combo_shape)
        self.sp_n_sides = QSpinBox(); self.sp_n_sides.setRange(3,12); self.sp_n_sides.setValue(5)
        t1l.addRow("N-gon sides:", self.sp_n_sides)
        self.combo_shape.currentTextChanged.connect(lambda t: self.sp_n_sides.setEnabled(t=="ngon"))
        self.sp_num_shapes = QSpinBox(); self.sp_num_shapes.setRange(5,500); self.sp_num_shapes.setValue(80)
        t1l.addRow("Shapes / individual:", self.sp_num_shapes)
        self.sp_alpha = QSpinBox(); self.sp_alpha.setRange(10,255); self.sp_alpha.setValue(128)
        t1l.addRow("Alpha:", self.sp_alpha)
        bg_info = QLabel("Canvas background auto-computed from target image average color")
        bg_info.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        bg_info.setWordWrap(True)
        t1l.addRow("", bg_info)
        tabs.addTab(t1, "🎨 Shapes")
        # Tab 2: GA
        t2 = QWidget(); t2l = QFormLayout(t2)
        self.sp_canvas_w = QSpinBox(); self.sp_canvas_w.setRange(32,512); self.sp_canvas_w.setValue(192)
        self.sp_canvas_h = QSpinBox(); self.sp_canvas_h.setRange(32,512); self.sp_canvas_h.setValue(192)
        self.sp_population = QSpinBox(); self.sp_population.setRange(10,300); self.sp_population.setValue(80)
        self.sp_mutation = QDoubleSpinBox(); self.sp_mutation.setRange(0.001,0.5); self.sp_mutation.setSingleStep(0.005); self.sp_mutation.setValue(0.02)
        self.sp_elite = QSpinBox(); self.sp_elite.setRange(1,50); self.sp_elite.setValue(4)
        self.sp_generations = QSpinBox(); self.sp_generations.setRange(10,10000); self.sp_generations.setValue(1000)
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 999999); self.sp_seed.setValue(0)
        self.sp_seed.setToolTip("0 = random, >0 = reproducible")
        for l,w in [("Canvas W",self.sp_canvas_w),("Canvas H",self.sp_canvas_h),
                     ("Population",self.sp_population),("Mutation",self.sp_mutation),
                     ("Elite",self.sp_elite),("Max gens",self.sp_generations),
                     ("Seed (0=random)",self.sp_seed)]:
            t2l.addRow(l,w)
        tabs.addTab(t2, "🧬 GA")
        # Tab 3: Advanced
        t3 = QWidget(); t3l = QVBoxLayout(t3)
        self.chk_smart_init = QCheckBox("Smart init"); self.chk_smart_init.setChecked(True)
        self.chk_hill_climb = QCheckBox("Hill climb"); self.chk_hill_climb.setChecked(True)
        hcr = QHBoxLayout(); hcr.addWidget(QLabel("Steps:"))
        self.sp_hill_climb_steps = QSpinBox(); self.sp_hill_climb_steps.setRange(0,200); self.sp_hill_climb_steps.setValue(20)
        hcr.addWidget(self.sp_hill_climb_steps)
        self.chk_adaptive = QCheckBox("Adaptive mutation"); self.chk_adaptive.setChecked(True)
        self.chk_ssim = QCheckBox("SSIM fitness")
        self.chk_hsv = QCheckBox("HSV color space")
        self.chk_multiscale = QCheckBox("Multi-scale")
        msr = QHBoxLayout(); msr.addWidget(QLabel("Ratio:"))
        self.sp_multiscale_ratio = QDoubleSpinBox(); self.sp_multiscale_ratio.setRange(0.1,0.7); self.sp_multiscale_ratio.setSingleStep(0.05); self.sp_multiscale_ratio.setValue(0.3)
        msr.addWidget(self.sp_multiscale_ratio)
        for w in [self.chk_smart_init,self.chk_hill_climb]: t3l.addWidget(w)
        t3l.addLayout(hcr)
        for w in [self.chk_adaptive,self.chk_ssim,self.chk_hsv,self.chk_multiscale]: t3l.addWidget(w)
        t3l.addLayout(msr)
        # Algorithm enhancements
        algo_group = QGroupBox("★ Algorithm Enhancements")
        agl = QVBoxLayout(algo_group)
        self.chk_residual = QCheckBox("Residual error-map guided placement"); self.chk_residual.setChecked(True)
        self.chk_annealing = QCheckBox("Simulated annealing acceptance"); self.chk_annealing.setChecked(True)
        self.chk_de_blend = QCheckBox("Differential evolution blending"); self.chk_de_blend.setChecked(True)
        self.chk_shape_split = QCheckBox("Shape split/merge operators"); self.chk_shape_split.setChecked(True)
        self.chk_dynamic_count = QCheckBox("Allow dynamic shape count"); self.chk_dynamic_count.setChecked(False)
        self.chk_fitness_cache = QCheckBox("Fitness cache"); self.chk_fitness_cache.setChecked(True)
        self.chk_diversity = QCheckBox("Diversity injection"); self.chk_diversity.setChecked(True)
        self.chk_importance = QCheckBox("Shape importance tracking"); self.chk_importance.setChecked(True)
        self.chk_gradient_mut = QCheckBox("Gradient-directed mutation"); self.chk_gradient_mut.setChecked(True)
        for w in [self.chk_residual, self.chk_annealing, self.chk_de_blend, self.chk_shape_split,
                  self.chk_dynamic_count, self.chk_fitness_cache, self.chk_diversity,
                  self.chk_importance, self.chk_gradient_mut]:
            agl.addWidget(w)
        t3l.addWidget(algo_group)
        # Performance
        perf_group = QGroupBox("★ Performance")
        pgl2 = QFormLayout(perf_group)
        self.sp_eval_scale = QDoubleSpinBox(); self.sp_eval_scale.setRange(0.25, 1.0)
        self.sp_eval_scale.setSingleStep(0.25); self.sp_eval_scale.setValue(0.5)
        self.sp_eval_scale.setToolTip("Evaluation resolution (lower = faster)")
        pgl2.addRow("Eval scale:", self.sp_eval_scale)
        t3l.addWidget(perf_group)
        t3l.addStretch()
        tabs.addTab(t3, "⚙ Advanced")
        # Tab 4: Timing
        t4 = QWidget(); t4l = QFormLayout(t4)
        self.sp_segment_duration = QDoubleSpinBox(); self.sp_segment_duration.setRange(1.0,120.0); self.sp_segment_duration.setValue(6.0)
        self.sp_fps = QSpinBox(); self.sp_fps.setRange(5,60); self.sp_fps.setValue(20)
        self.sp_display_w = QSpinBox(); self.sp_display_w.setRange(128,1280); self.sp_display_w.setValue(480)
        self.sp_display_h = QSpinBox(); self.sp_display_h.setRange(128,1280); self.sp_display_h.setValue(360)
        self.combo_stop_mode = QComboBox(); self.combo_stop_mode.addItems(TimingController.STOP_MODES)
        self.combo_stop_mode.setCurrentText("target_mse")
        self.sp_target_mse = QDoubleSpinBox(); self.sp_target_mse.setRange(0.0001, 0.5)
        self.sp_target_mse.setDecimals(4); self.sp_target_mse.setSingleStep(0.001); self.sp_target_mse.setValue(0.01)
        self.combo_transition = QComboBox(); self.combo_transition.addItems(TimingController.TRANSITION_TYPES)
        self.sp_transition_sec = QDoubleSpinBox(); self.sp_transition_sec.setRange(0,5); self.sp_transition_sec.setValue(0.5)
        self.sp_hold_sec = QDoubleSpinBox(); self.sp_hold_sec.setRange(0,10); self.sp_hold_sec.setValue(0.5)
        self.combo_speed_curve = QComboBox(); self.combo_speed_curve.addItems(TimingController.SPEED_CURVES)
        for l,w in [("Segment duration",self.sp_segment_duration),("FPS",self.sp_fps),
                     ("Display W",self.sp_display_w),("Display H",self.sp_display_h),
                     ("Stop mode",self.combo_stop_mode),("Target MSE",self.sp_target_mse),
                     ("Transition",self.combo_transition),("Transition sec",self.sp_transition_sec),
                     ("Hold final (sec)",self.sp_hold_sec),("Speed curve",self.combo_speed_curve)]:
            t4l.addRow(l,w)
        self.chk_intro = QCheckBox("Enable intro"); self.chk_intro.setChecked(False)
        self.intro_text = QLineEdit("Genetic Evolution")
        self.sp_intro_sec = QDoubleSpinBox(); self.sp_intro_sec.setRange(0.5,10); self.sp_intro_sec.setValue(2.0)
        self.chk_outro = QCheckBox("Enable outro"); self.chk_outro.setChecked(False)
        self.outro_text = QLineEdit("Generated with GA")
        self.sp_outro_sec = QDoubleSpinBox(); self.sp_outro_sec.setRange(0.5,10); self.sp_outro_sec.setValue(2.0)
        t4l.addRow("", self.chk_intro); t4l.addRow("Intro text:", self.intro_text); t4l.addRow("Intro sec:", self.sp_intro_sec)
        t4l.addRow("", self.chk_outro); t4l.addRow("Outro text:", self.outro_text); t4l.addRow("Outro sec:", self.sp_outro_sec)
        # Animation quality
        anim_group = QGroupBox("Animation Quality")
        agl2 = QFormLayout(anim_group)
        self.combo_camera_mode = QComboBox(); self.combo_camera_mode.addItems(["none", "subtle", "dynamic", "cinematic"])
        self.combo_camera_mode.setCurrentText("cinematic")
        agl2.addRow("Camera mode:", self.combo_camera_mode)
        self.chk_temporal_smooth = QCheckBox("Temporal smoothing"); self.chk_temporal_smooth.setChecked(True)
        agl2.addRow("", self.chk_temporal_smooth)
        self.sp_smooth_strength = QDoubleSpinBox(); self.sp_smooth_strength.setRange(0.0, 0.5)
        self.sp_smooth_strength.setSingleStep(0.05); self.sp_smooth_strength.setValue(0.3)
        agl2.addRow("Smoothing strength:", self.sp_smooth_strength)
        t4l.addRow("", anim_group)
        # Temporal coherence
        coh_group = QGroupBox("Temporal Coherence")
        cgl = QFormLayout(coh_group)
        self.sp_temporal_coherence = QDoubleSpinBox(); self.sp_temporal_coherence.setRange(0.0, 1.0)
        self.sp_temporal_coherence.setSingleStep(0.05); self.sp_temporal_coherence.setValue(0.3)
        cgl.addRow("Coherence strength:", self.sp_temporal_coherence)
        self.sp_inheritance = QDoubleSpinBox(); self.sp_inheritance.setRange(0.0, 1.0)
        self.sp_inheritance.setSingleStep(0.05); self.sp_inheritance.setValue(0.8)
        cgl.addRow("Inheritance ratio:", self.sp_inheritance)
        self.chk_progressive = QCheckBox("Progressive shape count"); self.chk_progressive.setChecked(False)
        cgl.addRow("", self.chk_progressive)
        self.chk_motion_blur = QCheckBox("Motion blur"); self.chk_motion_blur.setChecked(False)
        cgl.addRow("", self.chk_motion_blur)
        self.sp_blur_strength = QDoubleSpinBox(); self.sp_blur_strength.setRange(0.0, 0.7)
        self.sp_blur_strength.setSingleStep(0.05); self.sp_blur_strength.setValue(0.3)
        cgl.addRow("Blur strength:", self.sp_blur_strength)
        self.sp_keyframe_interval = QSpinBox(); self.sp_keyframe_interval.setRange(1, 20); self.sp_keyframe_interval.setValue(1)
        cgl.addRow("Keyframe interval:", self.sp_keyframe_interval)
        t4l.addRow("", coh_group)
        tabs.addTab(t4, "⏱ Timing")
        # Tab 5: Batch
        t5 = QWidget(); t5l = QVBoxLayout(t5)
        og = QGroupBox("Batch Output"); ogl = QFormLayout(og)
        odr = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./batch_output")
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._on_browse_output)
        odr.addWidget(self.output_dir_edit, 3); odr.addWidget(btn_browse)
        ogl.addRow("Output dir:", odr)
        self.name_template_edit = QLineEdit("{name}_{shape}")
        ogl.addRow("Name template:", self.name_template_edit)
        self.chk_export_png = QCheckBox("Export final PNG per image"); self.chk_export_png.setChecked(True)
        ogl.addRow("", self.chk_export_png)
        t5l.addWidget(og)
        sg = QGroupBox("Parameter Sweep"); sgl = QVBoxLayout(sg)
        self.preset_check_list = QListWidget(); self.preset_check_list.setMaximumHeight(120)
        for name in BUILTIN_PRESETS:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.preset_check_list.addItem(item)
        sgl.addWidget(self.preset_check_list)
        self.chk_sweep = QCheckBox("Run parameter sweep")
        sgl.addWidget(self.chk_sweep)
        t5l.addWidget(sg)
        pg = QGroupBox("Batch Progress"); pgl = QVBoxLayout(pg)
        self.progress_table = QTableWidget(0, 6)
        self.progress_table.setHorizontalHeaderLabels(["Image", "Status", "Gen", "MSE", "Time", "Output"])
        self.progress_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.progress_table.setMaximumHeight(150)
        self.progress_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pgl.addWidget(self.progress_table)
        t5l.addWidget(pg)
        t5l.addStretch()
        tabs.addTab(t5, "📦 Batch")
        # Tab 6: Post-processing
        t6 = QWidget(); t6l = QVBoxLayout(t6)
        qg = QGroupBox("Output Quality"); qgl = QFormLayout(qg)
        self.combo_quality = QComboBox()
        self.combo_quality.addItems(["web (720p, small)", "hd (1080p, balanced)"])
        self.combo_quality.setCurrentIndex(1)
        qgl.addRow("Quality tier:", self.combo_quality)
        self.chk_per_image_mp4 = QCheckBox("Generate per-image MP4 files"); self.chk_per_image_mp4.setChecked(True)
        qgl.addRow("", self.chk_per_image_mp4)
        self.chk_thumbnail = QCheckBox("Generate thumbnail images"); self.chk_thumbnail.setChecked(True)
        qgl.addRow("", self.chk_thumbnail)
        t6l.addWidget(qg)
        ppg = QGroupBox("Post-Processing"); ppgl = QFormLayout(ppg)
        self.sp_post_sharpen = QDoubleSpinBox(); self.sp_post_sharpen.setRange(0.0, 1.5)
        self.sp_post_sharpen.setSingleStep(0.1); self.sp_post_sharpen.setValue(0.3)
        ppgl.addRow("Sharpening:", self.sp_post_sharpen)
        self.combo_color_grade = QComboBox()
        self.combo_color_grade.addItems(["none", "cinematic", "warm", "cool", "vintage", "noir"])
        ppgl.addRow("Color grade:", self.combo_color_grade)
        self.sp_post_vignette = QDoubleSpinBox(); self.sp_post_vignette.setRange(0.0, 1.0)
        self.sp_post_vignette.setSingleStep(0.05); self.sp_post_vignette.setValue(0.2)
        ppgl.addRow("Vignette:", self.sp_post_vignette)
        self.sp_post_grain = QDoubleSpinBox(); self.sp_post_grain.setRange(0.0, 0.3)
        self.sp_post_grain.setSingleStep(0.01); self.sp_post_grain.setValue(0.0)
        ppgl.addRow("Film grain:", self.sp_post_grain)
        t6l.addWidget(ppg)
        cg = QGroupBox("Captions"); cgl2 = QFormLayout(cg)
        self.combo_caption_style = QComboBox()
        self.combo_caption_style.addItems(["none", "minimal", "cinematic", "bold"])
        cgl2.addRow("Caption style:", self.combo_caption_style)
        t6l.addWidget(cg)
        ag = QGroupBox("Audio"); agl3 = QFormLayout(ag)
        self.combo_audio_mood = QComboBox()
        self.combo_audio_mood.addItems(MusicGenerator.MOODS)
        self.combo_audio_mood.setCurrentText("ambient")
        agl3.addRow("Music mood:", self.combo_audio_mood)
        t6l.addWidget(ag)
        t6l.addStretch()
        tabs.addTab(t6, "🎬 Post")
        ll.addWidget(tabs, 3)
        # Preset bar
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["—"] + list(BUILTIN_PRESETS.keys()))
        self.combo_preset.currentTextChanged.connect(self._on_preset_combo)
        pr.addWidget(self.combo_preset, 2)
        btn_save_preset = QPushButton("💾 Save"); btn_save_preset.clicked.connect(self._save_preset)
        pr.addWidget(btn_save_preset)
        ll.addLayout(pr)
        # Action buttons
        ar2 = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start Batch"); self.btn_start.clicked.connect(self._on_start)
        self.btn_pause = QPushButton("⏸ Pause"); self.btn_pause.clicked.connect(self._on_pause); self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("■ Stop"); self.btn_stop.clicked.connect(self._on_stop); self.btn_stop.setEnabled(False)
        self.btn_next = QPushButton("Next ▶▶"); self.btn_next.clicked.connect(self._on_next)
        self.btn_snapshot = QPushButton("📷 Snap"); self.btn_snapshot.clicked.connect(self._on_snapshot)
        for b in (self.btn_start,self.btn_pause,self.btn_stop,self.btn_next,self.btn_snapshot): ar2.addWidget(b)
        ll.addLayout(ar2)
        self.progress = QProgressBar(); self.progress.setRange(0,100)
        ll.addWidget(self.progress)
        self.status_label = QLabel("Ready.")
        ll.addWidget(self.status_label)
        # Right panel
        right = QWidget(); rl = QVBoxLayout(right)
        preview_splitter = QSplitter(Qt.Vertical)
        self.preview_label = QLabel("Preview / Target")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480,320)
        self.preview_label.setStyleSheet("background:#1e1e1e;color:#aaa;border:1px solid #444;")
        preview_splitter.addWidget(self.preview_label)
        self.fitness_graph = FitnessGraphWidget()
        preview_splitter.addWidget(self.fitness_graph)
        preview_splitter.setSizes([320, 180])
        rl.addWidget(preview_splitter, 4)
        self.caption_label = QLabel(""); self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("color:#ddd;padding:6px;font-size:14px;")
        rl.addWidget(self.caption_label)
        log_g = QGroupBox("Log"); log_l = QVBoxLayout(log_g)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        log_l.addWidget(self.log_text); rl.addWidget(log_g, 2)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([500, 1000])

    def _build_statusbar(self):
        self.statusBar().showMessage("Ready")
        self.eta_label = QLabel("ETA: —"); self.eta_label.setStyleSheet("color:#aaa;padding:0 10px;")
        self.statusBar().addPermanentWidget(self.eta_label)

    def _setup_shortcuts(self):
        sc_pause = QAction(self); sc_pause.setShortcut(QKeySequence(Qt.Key_Space))
        sc_pause.triggered.connect(self._on_pause); self.addAction(sc_pause)
        sc_stop = QAction(self); sc_stop.setShortcut(QKeySequence(Qt.Key_Escape))
        sc_stop.triggered.connect(self._on_stop); self.addAction(sc_stop)
        sc_snap = QAction(self); sc_snap.setShortcut(QKeySequence("Ctrl+S"))
        sc_snap.triggered.connect(self._on_snapshot); self.addAction(sc_snap)
        sc_up = QAction(self); sc_up.setShortcut(QKeySequence("Ctrl+Up"))
        sc_up.triggered.connect(lambda: self._navigate_queue(-1)); self.addAction(sc_up)
        sc_down = QAction(self); sc_down.setShortcut(QKeySequence("Ctrl+Down"))
        sc_down.triggered.connect(lambda: self._navigate_queue(1)); self.addAction(sc_down)

    def _navigate_queue(self, delta):
        r = self.list_widget.currentRow()
        new_r = r + delta
        if 0 <= new_r < self.list_widget.count():
            self.list_widget.setCurrentRow(new_r)

    def _load_settings(self):
        geo = self.settings.value("geometry")
        if geo: self.restoreGeometry(geo)
        state = self.settings.value("windowState")
        if state: self.restoreState(state)
        last_preset = self.settings.value("last_preset", "Balanced")
        if last_preset in BUILTIN_PRESETS: self._load_preset(last_preset)

    def _save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _load_preset(self, name):
        p = BUILTIN_PRESETS.get(name)
        if not p: return
        self.combo_shape.setCurrentText(p.get("shape_mode","triangle"))
        self.sp_n_sides.setValue(p.get("n_sides",5))
        self.sp_num_shapes.setValue(p.get("num_shapes",80))
        self.sp_alpha.setValue(p.get("alpha",128))
        self.sp_canvas_w.setValue(p.get("canvas_w",192))
        self.sp_canvas_h.setValue(p.get("canvas_h",192))
        self.sp_population.setValue(p.get("population_size",80))
        self.sp_mutation.setValue(p.get("mutation_rate",0.02))
        self.sp_elite.setValue(p.get("elite_count",4))
        self.sp_generations.setValue(p.get("generations",1000))
        self.sp_seed.setValue(p.get("seed", 0))
        self.sp_segment_duration.setValue(p.get("segment_duration",6.0))
        self.sp_fps.setValue(p.get("fps",20))
        self.sp_display_w.setValue(p.get("display_w",480))
        self.sp_display_h.setValue(p.get("display_h",360))
        self.combo_stop_mode.setCurrentText(p.get("stop_mode","target_mse"))
        self.sp_target_mse.setValue(p.get("target_mse",0.01))
        self.sp_hill_climb_steps.setValue(p.get("hill_climb_steps",20))
        self.chk_smart_init.setChecked(p.get("use_smart_init",True))
        self.chk_hill_climb.setChecked(p.get("use_hill_climb",True))
        self.chk_adaptive.setChecked(p.get("use_adaptive_mutation",True))
        self.chk_ssim.setChecked(p.get("use_ssim",False))
        self.chk_hsv.setChecked(p.get("color_space","rgb")=="hsv")
        self.chk_multiscale.setChecked(p.get("use_multiscale",False))
        self.sp_multiscale_ratio.setValue(p.get("multiscale_ratio",0.3))
        self.combo_transition.setCurrentText(p.get("transition_type","cut"))
        self.sp_transition_sec.setValue(p.get("transition_seconds",0.5))
        self.sp_hold_sec.setValue(p.get("hold_seconds",0.5))
        self.combo_speed_curve.setCurrentText(p.get("speed_curve","linear"))
        self.chk_intro.setChecked(p.get("intro_enabled",False))
        self.chk_outro.setChecked(p.get("outro_enabled",False))
        self.chk_export_png.setChecked(p.get("export_png", True))
        self.chk_residual.setChecked(p.get("use_residual_guidance", True))
        self.chk_annealing.setChecked(p.get("use_annealing", True))
        self.chk_de_blend.setChecked(p.get("use_de_blend", True))
        self.chk_shape_split.setChecked(p.get("use_shape_split", True))
        self.chk_dynamic_count.setChecked(p.get("allow_dynamic_shape_count", False))
        self.chk_fitness_cache.setChecked(p.get("use_fitness_cache", True))
        self.chk_diversity.setChecked(p.get("use_diversity_injection", True))
        self.chk_importance.setChecked(p.get("use_importance_tracking", True))
        self.chk_gradient_mut.setChecked(p.get("use_gradient_mutation", True))
        self.sp_eval_scale.setValue(p.get("evaluation_scale", 0.5))
        self.combo_camera_mode.setCurrentText(p.get("camera_mode", "cinematic"))
        self.chk_temporal_smooth.setChecked(p.get("temporal_smooth", True))
        self.sp_smooth_strength.setValue(p.get("smooth_strength", 0.3))
        self.sp_temporal_coherence.setValue(p.get("temporal_coherence", 0.3))
        self.sp_inheritance.setValue(p.get("inheritance_strength", 0.8))
        self.chk_progressive.setChecked(p.get("progressive_shapes", False))
        self.chk_motion_blur.setChecked(p.get("motion_blur", False))
        self.sp_blur_strength.setValue(p.get("motion_blur_strength", 0.3))
        self.sp_keyframe_interval.setValue(p.get("keyframe_interval", 1))
        quality_idx = {"web": 0, "hd": 1}.get(p.get("quality_tier", "hd"), 1)
        self.combo_quality.setCurrentIndex(quality_idx)
        self.chk_per_image_mp4.setChecked(p.get("generate_per_image_mp4", True))
        self.chk_thumbnail.setChecked(p.get("generate_thumbnail", True))
        self.sp_post_sharpen.setValue(p.get("post_sharpen", 0.3))
        self.combo_color_grade.setCurrentText(p.get("post_color_grade", "none"))
        self.sp_post_vignette.setValue(p.get("post_vignette", 0.2))
        self.sp_post_grain.setValue(p.get("post_film_grain", 0.0))
        self.combo_caption_style.setCurrentText(p.get("caption_style", "none"))
        self.combo_audio_mood.setCurrentText(p.get("audio_mood", "ambient"))
        self.settings.setValue("last_preset", name)
        self._on_log(f"Loaded preset: {name}")

    def _on_preset_combo(self, text):
        if text != "—":
            self._load_preset(text); self.combo_preset.setCurrentIndex(0)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name: return
        BUILTIN_PRESETS[name] = self._collect_params()
        self.combo_preset.clear(); self.combo_preset.addItems(["—"] + list(BUILTIN_PRESETS.keys()))
        self._on_log(f"Saved preset: {name}")

    def _add_entry(self, url, caption="", add_thumbnail=True):
        entry = ImageEntry(url, caption)
        self.entries.append(entry)
        item = QListWidgetItem(caption or url)
        item.setData(Qt.UserRole, entry)
        self.list_widget.addItem(item)
        self._update_count()
        if add_thumbnail:
            threading.Thread(target=self._load_thumbnail, args=(len(self.entries)-1,), daemon=True).start()
        return entry

    def _load_thumbnail(self, idx):
        try:
            entry = self.entries[idx]
            img = ImageDownloader.download(entry.url, timeout=10, max_retries=2)
            img.thumbnail((64, 48))
            arr = np.asarray(img)
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            entry.thumbnail = pix; entry.preview_pixmap = pix
            if idx < self.list_widget.count():
                item = self.list_widget.item(idx)
                if item: item.setIcon(pix)
        except Exception: pass

    def _update_count(self):
        n = len(self.entries)
        self.count_label.setText(f"{n} image{'s' if n!=1 else ''}")

    def _load_sample_list(self):
        for url, cap in [
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/400px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg", "Mona Lisa"),
            ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Kandinsky_-_Jaune_Rouge_Bleu.jpg/400px-Kandinsky_-_Jaune_Rouge_Bleu.jpg", "Kandinsky"),
        ]: self._add_entry(url, cap)

    def _on_add_image(self):
        text = self.url_edit.text().strip()
        if not text: return
        caption = self.caption_edit.text().strip()
        if "{" in text and (".." in text or "," in text):
            urls = UrlPatternExpander.expand(text)
            for i, u in enumerate(urls):
                c = f"{caption} #{i+1}" if caption else ""
                self._add_entry(u, c)
            self._on_log(f"Expanded pattern → {len(urls)} URLs")
        else:
            self._add_entry(text, caption)
        self.url_edit.clear(); self.caption_edit.clear()

    def _on_remove_image(self):
        r = self.list_widget.currentRow()
        if r < 0: return
        self.list_widget.takeItem(r); self.entries.pop(r); self._update_count()

    def _on_clear_queue(self):
        if not self.entries: return
        if QMessageBox.question(self,"Clear",f"Remove {len(self.entries)} images?",
            QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:
            self.list_widget.clear(); self.entries.clear(); self._update_count()

    def _on_queue_selection_changed(self, row):
        if row < 0 or row >= len(self.entries): return
        entry = self.entries[row]
        self.caption_label.setText(entry.caption or "(no caption)")
        if entry.preview_pixmap:
            self.preview_label.setPixmap(entry.preview_pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_files_dropped(self, paths):
        total = 0
        for p in paths:
            try:
                if p.lower().endswith(".json"):
                    entries = JsonImageLoader.load_file(p)
                elif p.lower().endswith(".csv"):
                    entries = CsvImageLoader.load_file(p)
                elif os.path.isdir(p):
                    entries = self._load_images_from_folder(p)
                else: continue
                for e in entries: self._add_entry(e.url, e.caption)
                total += len(entries)
                self._on_log(f"Dropped: {os.path.basename(p)} ({len(entries)} images)")
            except Exception as ex: self._on_log(f"⚠ {os.path.basename(p)}: {ex}")
        if total: self.statusBar().showMessage(f"Dropped {total} images", 3000)

    def _on_load_json(self):
        paths, _ = QFileDialog.getOpenFileNames(self,"Select JSON","","JSON (*.json);;All (*)")
        if not paths: return
        t = 0
        for p in paths:
            try:
                ne = JsonImageLoader.load_file(p)
                for x in ne: self._add_entry(x.url, x.caption)
                t += len(ne); self._on_log(f"Loaded {len(ne)} from {os.path.basename(p)}")
            except Exception as ex: self._on_log(f"⚠ {os.path.basename(p)}: {ex}")
        QMessageBox.information(self,"Import",f"Added {t} images.")

    def _on_load_json_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"Select Folder")
        if not folder: return
        jf = sorted(f for f in os.listdir(folder) if f.lower().endswith(".json"))
        if not jf: QMessageBox.information(self,"","No .json files found."); return
        t = 0
        for fn in jf:
            try:
                ne = JsonImageLoader.load_file(os.path.join(folder,fn))
                for x in ne: self._add_entry(x.url, x.caption)
                t += len(ne)
            except Exception as ex: self._on_log(f"⚠ {fn}: {ex}")
        QMessageBox.information(self,"Import",f"Added {t} from {len(jf)} JSON files.")

    def _on_load_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if not folder: return
        entries = self._load_images_from_folder(folder)
        for e in entries: self._add_entry(e.url, e.caption)
        QMessageBox.information(self, "Import", f"Added {len(entries)} images.")

    def _load_images_from_folder(self, folder):
        entries = []
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
        files = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)
        for fn in files:
            full = os.path.join(folder, fn)
            url = Path(full).as_uri()
            caption = os.path.splitext(fn)[0]
            entries.append(ImageEntry(url, caption))
        return entries

    def _on_load_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select CSV", "", "CSV (*.csv);;All (*)")
        if not paths: return
        t = 0
        for p in paths:
            try:
                ne = CsvImageLoader.load_file(p)
                for x in ne: self._add_entry(x.url, x.caption)
                t += len(ne); self._on_log(f"Loaded {len(ne)} from {os.path.basename(p)}")
            except Exception as ex: self._on_log(f"⚠ {os.path.basename(p)}: {ex}")
        QMessageBox.information(self, "Import", f"Added {t} images from CSV.")

    def _on_expand_pattern(self):
        pattern, ok = QInputDialog.getText(self, "URL Pattern",
            "Enter URL pattern (e.g., https://example.com/img_{1..10}.jpg):")
        if not ok or not pattern: return
        urls = UrlPatternExpander.expand(pattern)
        if not urls: QMessageBox.information(self, "", "No URLs generated."); return
        caption_base, ok2 = QInputDialog.getText(self, "Caption",
            f"Generated {len(urls)} URLs.\nBase caption (optional):")
        if not ok2: return
        for i, u in enumerate(urls):
            c = f"{caption_base} #{i+1}" if caption_base else ""
            self._add_entry(u, c)
        self._on_log(f"Expanded pattern → {len(urls)} URLs")

    def _on_save_job(self):
        if not self.entries: QMessageBox.information(self, "", "Empty queue."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Job", "batch_job.json", "JSON (*.json)")
        if not path: return
        job = {"version": 3, "entries": [{"url": e.url, "caption": e.caption} for e in self.entries],
               "params": self._collect_params(), "output_dir": self.output_dir_edit.text(),
               "name_template": self.name_template_edit.text(), "saved_at": datetime.now().isoformat()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        self._on_log(f"💼 Job saved → {path}")

    def _on_load_job(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Job", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f:
                job = json.load(f)
            if "params" in job: self._restore_params(job["params"])
            if "output_dir" in job: self.output_dir_edit.setText(job["output_dir"])
            if "name_template" in job: self.name_template_edit.setText(job["name_template"])
            self.list_widget.clear(); self.entries.clear()
            for e in job.get("entries", []): self._add_entry(e["url"], e.get("caption", ""))
            self._on_log(f"💼 Job loaded: {len(self.entries)} images")
        except Exception as ex:
            QMessageBox.critical(self, "Error", f"Failed to load job: {ex}")

    def _restore_params(self, p):
        # Reuse load_preset logic by temporarily adding to presets
        BUILTIN_PRESETS["_job"] = p
        self._load_preset("_job")
        del BUILTIN_PRESETS["_job"]

    def _on_browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d: self.output_dir_edit.setText(d)

    def _collect_params(self):
        quality_map = {0: "web", 1: "hd"}
        return {
            "canvas_w": self.sp_canvas_w.value(), "canvas_h": self.sp_canvas_h.value(),
            "num_shapes": self.sp_num_shapes.value(), "population_size": self.sp_population.value(),
            "mutation_rate": self.sp_mutation.value(), "elite_count": self.sp_elite.value(),
            "alpha": self.sp_alpha.value(), "generations": self.sp_generations.value(),
            "segment_duration": self.sp_segment_duration.value(), "fps": self.sp_fps.value(),
            "display_w": self.sp_display_w.value(), "display_h": self.sp_display_h.value(),
            "n_sides": self.sp_n_sides.value(), "seed": self.sp_seed.value(),
            "shape_mode": self.combo_shape.currentText(),
            "use_smart_init": self.chk_smart_init.isChecked(),
            "use_hill_climb": self.chk_hill_climb.isChecked(),
            "hill_climb_steps": self.sp_hill_climb_steps.value(),
            "use_adaptive_mutation": self.chk_adaptive.isChecked(),
            "use_ssim": self.chk_ssim.isChecked(),
            "use_multiscale": self.chk_multiscale.isChecked(),
            "multiscale_ratio": self.sp_multiscale_ratio.value(),
            "color_space": "hsv" if self.chk_hsv.isChecked() else "rgb",
            "stop_mode": self.combo_stop_mode.currentText(),
            "target_mse": self.sp_target_mse.value(),
            "transition_type": self.combo_transition.currentText(),
            "transition_seconds": self.sp_transition_sec.value(),
            "hold_seconds": self.sp_hold_sec.value(),
            "speed_curve": self.combo_speed_curve.currentText(),
            "intro_enabled": self.chk_intro.isChecked(),
            "intro_text": self.intro_text.text(),
            "intro_seconds": self.sp_intro_sec.value(),
            "outro_enabled": self.chk_outro.isChecked(),
            "outro_text": self.outro_text.text(),
            "outro_seconds": self.sp_outro_sec.value(),
            "export_png": self.chk_export_png.isChecked(),
            "use_residual_guidance": self.chk_residual.isChecked(),
            "use_multiscale_fitness": self.chk_multiscale.isChecked() and self.chk_ssim.isChecked(),
            "use_annealing": self.chk_annealing.isChecked(),
            "use_de_blend": self.chk_de_blend.isChecked(),
            "use_shape_split": self.chk_shape_split.isChecked(),
            "allow_dynamic_shape_count": self.chk_dynamic_count.isChecked(),
            "use_fitness_cache": self.chk_fitness_cache.isChecked(),
            "use_diversity_injection": self.chk_diversity.isChecked(),
            "use_importance_tracking": self.chk_importance.isChecked(),
            "use_gradient_mutation": self.chk_gradient_mut.isChecked(),
            "evaluation_scale": self.sp_eval_scale.value(),
            "camera_mode": self.combo_camera_mode.currentText(),
            "temporal_smooth": self.chk_temporal_smooth.isChecked(),
            "smooth_strength": self.sp_smooth_strength.value(),
            "temporal_coherence": self.sp_temporal_coherence.value(),
            "inheritance_strength": self.sp_inheritance.value(),
            "progressive_shapes": self.chk_progressive.isChecked(),
            "motion_blur": self.chk_motion_blur.isChecked(),
            "motion_blur_strength": self.sp_blur_strength.value(),
            "keyframe_interval": self.sp_keyframe_interval.value(),
            "quality_tier": quality_map.get(self.combo_quality.currentIndex(), "hd"),
            "generate_per_image_mp4": self.chk_per_image_mp4.isChecked(),
            "generate_thumbnail": self.chk_thumbnail.isChecked(),
            "post_sharpen": self.sp_post_sharpen.value(),
            "post_color_grade": self.combo_color_grade.currentText(),
            "post_vignette": self.sp_post_vignette.value(),
            "post_film_grain": self.sp_post_grain.value(),
            "caption_style": self.combo_caption_style.currentText(),
            "audio_mood": self.combo_audio_mood.currentText(),
        }

    def _on_start(self):
        if not self.entries: QMessageBox.warning(self, "", "Add images first."); return
        output_dir = self.output_dir_edit.text().strip() or "./batch_output"
        name_template = self.name_template_edit.text().strip() or "{name}_{shape}"
        sweep_presets = []
        if self.chk_sweep.isChecked():
            for i in range(self.preset_check_list.count()):
                item = self.preset_check_list.item(i)
                if item.checkState() == Qt.Checked: sweep_presets.append(item.text())
        if sweep_presets:
            self._start_sweep(sweep_presets, output_dir, name_template)
        else:
            self._start_batch(output_dir, name_template, "custom", self._collect_params())

    def _start_sweep(self, preset_names, output_dir, name_template):
        total_runs = len(self.entries) * len(preset_names)
        reply = QMessageBox.question(self, "Parameter Sweep",
            f"Run {len(self.entries)} images × {len(preset_names)} presets = {total_runs} runs.\nContinue?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self._sweep_state = {"presets": preset_names, "preset_idx": 0,
                              "output_dir": output_dir, "name_template": name_template}
        self._run_next_sweep_preset()

    def _run_next_sweep_preset(self):
        state = self._sweep_state
        if state["preset_idx"] >= len(state["presets"]):
            self._on_log("=== Sweep complete ===")
            QMessageBox.information(self, "Done", "Parameter sweep complete!"); return
        preset_name = state["presets"][state["preset_idx"]]
        preset_params = BUILTIN_PRESETS[preset_name].copy()
        for k in ("segment_duration", "fps", "display_w", "display_h"):
            preset_params[k] = self._collect_params()[k]
        self._on_log(f"\n=== Sweep {state['preset_idx']+1}/{len(state['presets'])}: {preset_name} ===")
        self._start_batch(state["output_dir"], state["name_template"], preset_name, preset_params,
                          sweep_callback=lambda: self._on_sweep_preset_done())

    def _on_sweep_preset_done(self):
        self._sweep_state["preset_idx"] += 1
        QTimer.singleShot(500, self._run_next_sweep_preset)

    def _start_batch(self, output_dir, name_template, preset_name, params, sweep_callback=None):
        self._sweep_callback = sweep_callback
        for e in self.entries:
            e.status = "pending"; e.current_gen = 0; e.current_mse = 0.0
            e.output_path = None; e.png_path = None; e.started_at = None; e.finished_at = None
        self.progress_table.setRowCount(len(self.entries))
        for i, e in enumerate(self.entries):
            self.progress_table.setItem(i, 0, QTableWidgetItem(e.caption or Path(e.url).stem[:30]))
            self.progress_table.setItem(i, 1, QTableWidgetItem("pending"))
            self.progress_table.setItem(i, 2, QTableWidgetItem("0"))
            self.progress_table.setItem(i, 3, QTableWidgetItem("—"))
            self.progress_table.setItem(i, 4, QTableWidgetItem("—"))
            self.progress_table.setItem(i, 5, QTableWidgetItem(""))
        self.fitness_graph.clear()
        self.fitness_graph.title = f"Fitness — {params.get('shape_mode', '')}"
        self.worker_thread = QThread()
        self.worker = BatchWorker(list(self.entries), params, output_dir, name_template, preset_name)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.image_started.connect(self._on_image_started)
        self.worker.image_finished.connect(self._on_image_finished)
        self.worker.image_failed.connect(self._on_image_failed)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._on_log)
        self.worker.eta_update.connect(self._on_eta)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.image_progress.connect(self._on_image_progress)
        self.worker.fitness_point.connect(self._on_fitness_point)
        self.worker.all_done.connect(self._cleanup_worker)
        self.worker.error.connect(self._cleanup_worker)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_next.setEnabled(False); self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Pause")
        self.progress.setValue(0)
        self.worker_thread.start()

    def _cleanup_worker(self, *args):
        if self.worker_thread: self.worker_thread.quit(); self.worker_thread.wait()
        self.worker = self.worker_thread = None
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_next.setEnabled(True); self.btn_pause.setEnabled(False)
        self.eta_label.setText("ETA: —")
        if hasattr(self, '_sweep_callback') and self._sweep_callback:
            cb = self._sweep_callback; self._sweep_callback = None; cb()

    def _on_stop(self):
        if self.worker: self.worker.request_stop(); self._on_log("Stop requested…")

    def _on_pause(self):
        if not self.worker: return
        if self.btn_pause.text().startswith("⏸"):
            self.worker.request_pause()
            self.btn_pause.setText("▶ Resume"); self._on_log("⏸ Paused")
        else:
            self.worker.request_resume()
            self.btn_pause.setText("⏸ Pause"); self._on_log("▶ Resumed")

    def _on_next(self):
        r = self.list_widget.currentRow()
        if r < 0: QMessageBox.information(self, "", "Select an image."); return
        output_dir = self.output_dir_edit.text().strip() or "./batch_output"
        name_template = self.name_template_edit.text().strip() or "{name}_{shape}"
        self._start_batch(output_dir, name_template, "custom", self._collect_params())

    def _on_snapshot(self):
        if self._current_frame is None:
            QMessageBox.information(self, "", "No frame to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "snapshot.png", "PNG (*.png)")
        if not path: return
        Image.fromarray(self._current_frame).save(path)
        self._on_log(f"📷 Snapshot saved → {path}")

    def _on_frame(self, img_idx, gen, mse, frame_rgb):
        self._current_frame = frame_rgb.copy()
        h, w, _ = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, 3*w, QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.status_label.setText(f"Image {img_idx+1}/{len(self.entries)}  |  Gen {gen}  |  MSE {mse:.5f}")

    def _on_image_started(self, idx, caption):
        self.caption_label.setText(caption or "(no caption)")
        self._on_log(f"▶ Image {idx+1}: {caption or '(no caption)'}")
        if idx < self.list_widget.count(): self.list_widget.setCurrentRow(idx)
        if idx < self.progress_table.rowCount():
            self.progress_table.setItem(idx, 1, QTableWidgetItem("running"))
        self.fitness_graph.clear()
        self.fitness_graph.title = f"Fitness — {caption or f'Image {idx+1}'}"

    def _on_image_finished(self, idx, output_path, final_mse, gens_run, png_path):
        self._on_log(f"✓ Image {idx+1} complete → {os.path.basename(output_path)}")
        if idx < self.progress_table.rowCount():
            self.progress_table.setItem(idx, 1, QTableWidgetItem("✓ done"))
            self.progress_table.setItem(idx, 2, QTableWidgetItem(f"{gens_run}"))
            self.progress_table.setItem(idx, 3, QTableWidgetItem(f"{final_mse:.5f}"))
            entry = self.entries[idx] if idx < len(self.entries) else None
            if entry and entry.started_at and entry.finished_at:
                t = entry.finished_at - entry.started_at
                self.progress_table.setItem(idx, 4, QTableWidgetItem(f"{t:.1f}s"))
            self.progress_table.setItem(idx, 5, QTableWidgetItem(os.path.basename(output_path)))

    def _on_image_failed(self, idx, error):
        if idx < self.progress_table.rowCount():
            self.progress_table.setItem(idx, 1, QTableWidgetItem("✗ failed"))
            self.progress_table.setItem(idx, 5, QTableWidgetItem(error[:40]))

    def _on_image_progress(self, idx, current_gen, total_gens, mse):
        if idx < self.progress_table.rowCount():
            self.progress_table.setItem(idx, 2, QTableWidgetItem(f"{current_gen}/{total_gens}"))
            self.progress_table.setItem(idx, 3, QTableWidgetItem(f"{mse:.5f}"))

    def _on_fitness_point(self, img_idx, gen, mse):
        self.fitness_graph.add_point(gen, mse)

    def _on_all_done(self, output_dir, succeeded, failed, stats_path):
        self.progress.setValue(100)
        self.status_label.setText(f"Done: {succeeded} succeeded, {failed} failed → {output_dir}")
        self.statusBar().showMessage(f"Batch complete: {output_dir}", 10000)
        QMessageBox.information(self, "Batch Done",
            f"Succeeded: {succeeded}\nFailed: {failed}\nOutput: {output_dir}\n\nStats: {stats_path}")

    def _on_error(self, msg):
        self._on_log(msg); QMessageBox.critical(self, "Error", msg)

    def _on_log(self, msg):
        self.log_text.append(msg)

    def _on_eta(self, text):
        self.eta_label.setText(text)

    def _on_progress(self, pct):
        self.progress.setValue(pct)

    def _on_export_json(self):
        if not self.entries: QMessageBox.information(self, "", "Empty queue."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export Queue", "queue.json", "JSON (*.json)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"url": e.url, "caption": e.caption} for e in self.entries], f, indent=2)
        self._on_log(f"Exported → {path}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()