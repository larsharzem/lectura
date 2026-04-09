#!/usr/bin/env python3
"""
Assemble a karaoke-style video from:
  - Audio file (lecture recording)
  - Word-level timestamps (from transcribe_words.py)
  - Prompts with semantic chunk boundaries (from generate_prompts.py)
  - Generated images OR animated backgrounds

Background modes:
  images   - Generated illustrations (default, requires --images)
  lava     - Animated abstract blobs, lava lamp style
  waveform - Animated frequency spectrum bar chart
  radial   - Circular spectrum ring radiating from center

Usage:
    python assemble_video.py \\
        --audio lecture.mp3 --words words.json --prompts prompts.jsonl \\
        --images ~/ComfyUI/output/ --output video.mp4

    python assemble_video.py \\
        --audio lecture.mp3 --words words.json --prompts prompts.jsonl \\
        --bg-mode lava --colors "#1a1a2e,#16213e,#0f3460,#e94560" --output video.mp4

    python assemble_video.py \\
        --audio lecture.mp3 --words words.json --prompts prompts.jsonl \\
        --bg-mode waveform --colors "#0d1b2a,#1b263b,#415a77,#778da9" --output video.mp4

    python assemble_video.py \\
        --audio lecture.mp3 --words words.json --prompts prompts.jsonl \\
        --bg-mode radial --colors "#0a0a0a,#1a1a2e,#e94560,#0f3460,#16213e" --output video.mp4

Install:
    pip install numpy Pillow

Requires: ffmpeg installed and on PATH
"""

import argparse
import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip install Pillow numpy")
    sys.exit(1)


# ─── Configuration ──────────────────────────────────────────────────────────

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30

# Text overlay settings
TEXT_AREA_HEIGHT = 280          # pixels reserved for text at bottom
TEXT_MARGIN_X = 80              # horizontal margin
TEXT_MARGIN_BOTTOM = 40         # from bottom edge
FONT_SIZE = 36
FONT_COLOR = (255, 255, 255)    # white
HIGHLIGHT_BG = (180, 180, 180, 180)  # semi-transparent gray background for current word
SHADOW_COLOR = (0, 0, 0)       # black shadow for readability
BG_OVERLAY_ALPHA = 0            # text bar background transparency (0 = fully transparent)
HIGHLIGHT_PAD_X = 4             # horizontal padding around highlight box
HIGHLIGHT_PAD_Y = 2             # vertical padding around highlight box
HIGHLIGHT_RADIUS = 6            # corner radius for rounded highlight
MAX_WORDS_PER_PAGE = 15         # max words shown at once (pages break here even within a chunk)

# Try to use a good system font, fall back gracefully
FONT_CANDIDATES = [
    "./Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a TrueType font, fall back to default."""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Fallback
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, IOError):
        print("WARNING: No TrueType font found. Text rendering will be basic.")
        return ImageFont.load_default()


def get_bold_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a bold font variant."""
    bold_candidates = [
        p.replace("Regular", "Bold").replace("Sans.", "Sans-Bold.")
        for p in FONT_CANDIDATES
    ] + [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in bold_candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return get_font(size)


# ─── Color Palette ──────────────────────────────────────────────────────────

DEFAULT_COLORS = ["#1a1a2e", "#16213e", "#0f3460", "#e94560"]

def parse_colors(color_str: str) -> list[tuple]:
    """Parse comma-separated hex colors into list of (R, G, B) tuples."""
    colors = []
    for c in color_str.split(","):
        c = c.strip().lstrip("#")
        if len(c) == 6:
            colors.append((int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)))
        else:
            print(f"WARNING: Invalid color '{c}', skipping")
    return colors if colors else [(26, 26, 46), (22, 33, 62), (15, 52, 96), (233, 69, 96)]


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGB colors."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def palette_sample(colors: list[tuple], t: float) -> tuple:
    """Sample a color from the palette at position t (0.0–1.0)."""
    t = max(0.0, min(1.0, t))
    n = len(colors)
    if n == 1:
        return colors[0]
    idx = t * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return lerp_color(colors[lo], colors[hi], frac)


# ─── Background Generators ─────────────────────────────────────────────────

def load_audio_samples(audio_path: str, sample_rate: int = 22050) -> tuple:
    """Extract mono audio samples via ffmpeg for spectrum visualization."""
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-ac", "1",                    # mono
        "-ar", str(sample_rate),       # resample
        "-f", "s16le",                 # raw 16-bit signed PCM
        "-v", "quiet",
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    # Normalize to -1.0 .. 1.0
    peak = np.max(np.abs(samples)) or 1.0
    return samples / peak, sample_rate


def generate_lava_bg(time_sec: float, colors: list[tuple]) -> np.ndarray:
    """
    Generate an animated lava lamp background frame.
    Uses layered 2D sine/cosine waves at different frequencies and speeds
    to create smooth, evolving organic blobs.
    """
    # Work at quarter resolution for speed, then upscale
    h4 = VIDEO_HEIGHT // 4
    w4 = VIDEO_WIDTH // 4

    # Create coordinate grids (normalized 0..1)
    ys = np.linspace(0, 1, h4, dtype=np.float32)
    xs = np.linspace(0, 1, w4, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    t = time_sec

    # Layer multiple sine fields with different frequencies and phase speeds
    field = np.zeros((h4, w4), dtype=np.float32)

    # Each layer: sin(freq_x * x + phase_x(t)) * sin(freq_y * y + phase_y(t))
    layers = [
        (2.1, 1.7, 0.15, 0.19),   # (freq_x, freq_y, speed_x, speed_y)
        (3.3, 2.5, -0.23, 0.13),
        (1.5, 3.1, 0.11, -0.17),
        (4.1, 1.9, -0.09, 0.21),
        (2.7, 3.7, 0.17, -0.11),
    ]

    for fx, fy, sx, sy in layers:
        wave = (
            np.sin(fx * np.pi * 2 * xg + t * sx * np.pi * 2) *
            np.sin(fy * np.pi * 2 * yg + t * sy * np.pi * 2)
        )
        field += wave

    # Add a slow radial pulse
    cx = 0.5 + 0.15 * math.sin(t * 0.13)
    cy = 0.5 + 0.15 * math.cos(t * 0.11)
    dist = np.sqrt((xg - cx)**2 + (yg - cy)**2)
    field += 0.8 * np.sin(dist * 6.0 - t * 0.3)

    # Normalize to 0..1
    fmin, fmax = field.min(), field.max()
    if fmax - fmin > 0:
        field = (field - fmin) / (fmax - fmin)
    else:
        field = np.full_like(field, 0.5)

    # Map field values to palette colors
    frame_small = np.zeros((h4, w4, 3), dtype=np.uint8)
    n_colors = len(colors)
    for i in range(n_colors - 1):
        lo = i / (n_colors - 1)
        hi = (i + 1) / (n_colors - 1)
        mask = (field >= lo) & (field < hi)
        t_local = (field[mask] - lo) / (hi - lo) if (hi - lo) > 0 else field[mask] * 0

        c1, c2 = colors[i], colors[i + 1]
        for ch in range(3):
            frame_small[:, :, ch][mask] = (
                c1[ch] + (c2[ch] - c1[ch]) * t_local
            ).astype(np.uint8)

    # Handle edge: field == 1.0
    mask_top = field >= 1.0
    for ch in range(3):
        frame_small[:, :, ch][mask_top] = colors[-1][ch]

    # Upscale to full resolution with smooth interpolation
    img = Image.fromarray(frame_small, "RGB")
    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.BILINEAR)
    return img


def generate_spectrum_bg(time_sec: float, audio_samples: np.ndarray,
                         sample_rate: int, colors: list[tuple],
                         prev_state: dict | None = None) -> tuple:
    """
    Generate an FFT-based spectrum analyzer background.
    Returns (PIL Image, state dict for next frame).

    Uses per-band adaptive normalization so all frequency ranges
    show proportional activity.
    First color in palette is the background; remaining colors are used for bars.
    """
    bg_color = colors[0] if colors else (10, 10, 20)
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)

    total_samples = len(audio_samples)
    center_sample = int(time_sec * sample_rate)

    # FFT window size — larger = better frequency resolution
    fft_size = 4096
    half_fft = fft_size // 2

    # Extract window centered on current time
    win_start = max(0, center_sample - half_fft)
    win_end = min(total_samples, center_sample + half_fft)
    window = np.zeros(fft_size, dtype=np.float32)

    if win_end > win_start:
        chunk = audio_samples[win_start:win_end]
        offset = max(0, half_fft - center_sample)
        window[offset:offset + len(chunk)] = chunk

    # Apply Hann window to reduce spectral leakage
    hann = np.hanning(fft_size).astype(np.float32)
    window *= hann

    # Compute FFT magnitude (only positive frequencies)
    spectrum = np.abs(np.fft.rfft(window))

    # Group into bars with logarithmic frequency spacing
    num_bars = 128
    bar_colors = colors[1:] if len(colors) > 1 else colors

    freq_min = 60
    freq_max = sample_rate / 2
    bin_edges = np.logspace(
        np.log10(freq_min), np.log10(freq_max),
        num_bars + 1
    )

    freq_per_bin = sample_rate / fft_size
    bar_values = np.zeros(num_bars, dtype=np.float32)

    for i in range(num_bars):
        lo_bin = int(bin_edges[i] / freq_per_bin)
        hi_bin = int(bin_edges[i + 1] / freq_per_bin)
        lo_bin = max(0, min(lo_bin, len(spectrum) - 1))
        hi_bin = max(lo_bin + 1, min(hi_bin, len(spectrum)))
        bar_values[i] = np.mean(spectrum[lo_bin:hi_bin])

    # Per-band adaptive normalization
    if prev_state is None:
        prev_state = {
            "spectrum": np.zeros(num_bars, dtype=np.float32),
            "band_peaks": np.full(num_bars, 1e-10, dtype=np.float32),
        }

    band_peaks = prev_state["band_peaks"]
    peak_decay = 0.995
    band_peaks = np.maximum(band_peaks * peak_decay, bar_values)
    band_peaks = np.maximum(band_peaks, 1e-10)

    bar_values = bar_values / band_peaks

    # Gentle power curve and scale
    bar_values = np.power(bar_values, 1.2)
    bar_values *= 0.75

    # Asymmetric smoothing: fast attack, slow decay
    prev_spectrum = prev_state["spectrum"]
    if len(prev_spectrum) == num_bars:
        attack = 0.15
        decay = 0.75
        blend = np.where(bar_values >= prev_spectrum, attack, decay)
        bar_values = blend * prev_spectrum + (1 - blend) * bar_values

    # Build state for next frame
    next_state = {
        "spectrum": bar_values.copy(),
        "band_peaks": band_peaks,
    }

    # Render bars — edge to edge, growing upward from just above the text bar
    bar_area_bottom = VIDEO_HEIGHT - TEXT_AREA_HEIGHT
    bar_area_top = 0
    max_bar_h = bar_area_bottom - bar_area_top

    total_bar_width = VIDEO_WIDTH
    bar_width = max(2, total_bar_width // num_bars - 2)
    bar_gap = max(1, (total_bar_width - bar_width * num_bars) // max(num_bars - 1, 1))
    bar_step = bar_width + bar_gap
    x_offset = (VIDEO_WIDTH - bar_step * num_bars + bar_gap) // 2

    for i in range(num_bars):
        amp = float(np.clip(bar_values[i], 0.0, 1.0))
        bar_h = int(amp * max_bar_h)
        if bar_h < 1:
            continue

        # Color from palette based on bar position
        t_pos = i / max(num_bars - 1, 1)
        bar_color = palette_sample(bar_colors, t_pos)

        # Brighten based on amplitude
        brightness = 0.5 + 0.5 * amp
        bar_color = tuple(min(255, int(c * brightness)) for c in bar_color)

        x = x_offset + i * bar_step
        y_top = max(0, bar_area_bottom - bar_h)
        draw.rectangle(
            [(x, y_top), (x + bar_width, bar_area_bottom)],
            fill=bar_color
        )

    return img, next_state


def generate_radial_bg(time_sec: float, audio_samples: np.ndarray,
                       sample_rate: int, colors: list[tuple],
                       prev_state: dict | None = None) -> tuple:
    """
    Generate a circular spectrum analyzer background.
    A circle sits in the center; frequency bars radiate outward from it.
    Returns (PIL Image, state dict for next frame).

    prev_state carries both smoothed bar values and per-band peak tracking
    for independent normalization per frequency band.

    First color = background, second color = inner circle, rest = bar gradient.
    """
    # Render at 2x resolution for antialiasing, then downscale
    SS = 2
    w2 = VIDEO_WIDTH * SS
    h2 = VIDEO_HEIGHT * SS

    bg_color = colors[0] if colors else (10, 10, 20)
    img = Image.new("RGB", (w2, h2), bg_color)
    draw = ImageDraw.Draw(img)

    total_samples = len(audio_samples)
    center_sample = int(time_sec * sample_rate)

    # FFT
    fft_size = 4096
    half_fft = fft_size // 2

    win_start = max(0, center_sample - half_fft)
    win_end = min(total_samples, center_sample + half_fft)
    window = np.zeros(fft_size, dtype=np.float32)

    if win_end > win_start:
        chunk = audio_samples[win_start:win_end]
        offset = max(0, half_fft - center_sample)
        window[offset:offset + len(chunk)] = chunk

    hann = np.hanning(fft_size).astype(np.float32)
    window *= hann
    spectrum = np.abs(np.fft.rfft(window))

    # Group into bars
    num_bars = 128
    freq_min = 60
    freq_max = sample_rate / 2
    bin_edges = np.logspace(
        np.log10(freq_min), np.log10(freq_max),
        num_bars + 1
    )
    freq_per_bin = sample_rate / fft_size
    bar_values = np.zeros(num_bars, dtype=np.float32)

    for i in range(num_bars):
        lo_bin = int(bin_edges[i] / freq_per_bin)
        hi_bin = int(bin_edges[i + 1] / freq_per_bin)
        lo_bin = max(0, min(lo_bin, len(spectrum) - 1))
        hi_bin = max(lo_bin + 1, min(hi_bin, len(spectrum)))
        bar_values[i] = np.mean(spectrum[lo_bin:hi_bin])

    # Per-band normalization: each bar is measured against its own
    # recent peak, so quiet frequency bands still show activity
    if prev_state is None:
        prev_state = {
            "spectrum": np.zeros(num_bars, dtype=np.float32),
            "band_peaks": np.full(num_bars, 1e-10, dtype=np.float32),
        }

    band_peaks = prev_state["band_peaks"]

    # Update per-band peaks with slow decay — each band remembers
    # its own recent maximum and decays it gradually
    peak_decay = 0.995  # slow decay: peaks fade over ~6 seconds at 30fps
    band_peaks = np.maximum(band_peaks * peak_decay, bar_values)
    # Ensure no division by zero
    band_peaks = np.maximum(band_peaks, 1e-10)

    # Normalize each bar against its own peak
    bar_values = bar_values / band_peaks

    # Gentle power curve and scale
    bar_values = np.power(bar_values, 1.2)
    bar_values *= 0.75

    # Asymmetric smoothing: fast attack, slow decay
    prev_spectrum = prev_state["spectrum"]
    if len(prev_spectrum) == num_bars:
        attack = 0.15
        decay = 0.75
        blend = np.where(bar_values >= prev_spectrum, attack, decay)
        bar_values = blend * prev_spectrum + (1 - blend) * bar_values

    # Build state for next frame
    next_state = {
        "spectrum": bar_values.copy(),
        "band_peaks": band_peaks,
    }

    # Geometry — all in 2x space
    cx = w2 / 2
    cy = (h2 - TEXT_AREA_HEIGHT * SS) / 2  # center in the area above text
    # Inner circle radius — proportional to the smaller dimension
    usable_h = h2 - TEXT_AREA_HEIGHT * SS
    ring_space = min(w2, usable_h)
    inner_radius = ring_space * 0.12
    max_bar_length = ring_space * 0.44  # max length a bar can extend outward
    bar_angular_width = 0.7  # fraction of the angular slot each bar fills (vs gap)

    # Draw inner circle — outline only, no fill
    r = inner_radius
    draw.ellipse(
        [(cx - r, cy - r), (cx + r, cy + r)],
        outline=(200, 200, 200),
        width=2 * SS
    )

    # Draw bars radiating outward — use all non-background colors
    bar_colors = colors[1:] if len(colors) > 1 else colors
    if not bar_colors:
        bar_colors = [(200, 200, 200)]

    angle_step = 2 * math.pi / num_bars
    half_width_angle = angle_step * bar_angular_width / 2

    for i in range(num_bars):
        amp = float(np.clip(bar_values[i], 0.0, 1.0))
        if amp < 0.01:
            continue

        bar_length = amp * max_bar_length
        angle = i * angle_step - math.pi / 2  # start from top

        # Inner and outer radius for this bar
        r_inner = inner_radius + 2
        r_outer = inner_radius + 2 + bar_length

        # Four corners of the bar as a trapezoid
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        cos_l = math.cos(angle - half_width_angle)
        sin_l = math.sin(angle - half_width_angle)
        cos_r = math.cos(angle + half_width_angle)
        sin_r = math.sin(angle + half_width_angle)

        points = [
            (cx + cos_l * r_inner, cy + sin_l * r_inner),
            (cx + cos_r * r_inner, cy + sin_r * r_inner),
            (cx + cos_r * r_outer, cy + sin_r * r_outer),
            (cx + cos_l * r_outer, cy + sin_l * r_outer),
        ]

        # Color from palette
        t_pos = i / max(num_bars - 1, 1)
        bar_color = palette_sample(bar_colors, t_pos)

        brightness = 0.5 + 0.5 * amp
        bar_color = tuple(min(255, int(c * brightness)) for c in bar_color)

        draw.polygon(points, fill=bar_color)

    # Downscale to target resolution — this produces antialiased edges
    img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.LANCZOS)

    return img, next_state


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_words(words_path: str) -> list[dict]:
    """Load word-level timestamps from transcribe_words.py output."""
    with open(words_path) as f:
        data = json.load(f)
    return data["words"]


def load_prompts(prompts_path: str) -> list[dict]:
    """Load semantic chunk info from generate_prompts.py output."""
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def find_images(images_dir: str, prompts: list[dict]) -> dict:
    """
    Map chunk index → image file path.
    Looks for files matching the pattern from batch_generate.py: 0000_00001_.png etc.
    """
    img_dir = Path(images_dir)
    mapping = {}

    for prompt in prompts:
        idx = prompt["index"]
        prefix = f"{idx:04d}"

        # ComfyUI saves as {prefix}_{counter}_.png
        candidates = sorted(img_dir.glob(f"{prefix}*"))
        if candidates:
            mapping[idx] = str(candidates[0])
        else:
            # Also try without underscore
            candidates = sorted(img_dir.glob(f"{prefix}.*"))
            if candidates:
                mapping[idx] = str(candidates[0])

    return mapping


def get_chunk_for_time(prompts: list[dict], time_sec: float) -> int | None:
    """Find which semantic chunk a given timestamp falls into."""
    for p in prompts:
        if p["start_sec"] <= time_sec < p["end_sec"]:
            return p["index"]
    # If past the last chunk, use the last one
    if prompts and time_sec >= prompts[-1]["start_sec"]:
        return prompts[-1]["index"]
    return None


# ─── Frame Rendering ────────────────────────────────────────────────────────

def load_and_resize_image(path: str) -> Image.Image:
    """Load image and resize/crop to video dimensions."""
    img = Image.open(path).convert("RGB")

    # Scale to cover the video frame, then center crop
    img_ratio = img.width / img.height
    vid_ratio = VIDEO_WIDTH / VIDEO_HEIGHT

    if img_ratio > vid_ratio:
        # Image is wider — scale by height, crop width
        new_height = VIDEO_HEIGHT
        new_width = int(img.width * (VIDEO_HEIGHT / img.height))
    else:
        # Image is taller — scale by width, crop height
        new_width = VIDEO_WIDTH
        new_height = int(img.height * (VIDEO_WIDTH / img.width))

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Center crop
    left = (new_width - VIDEO_WIDTH) // 2
    top = (new_height - VIDEO_HEIGHT) // 2
    img = img.crop((left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT))

    return img


def assign_words_to_chunks(words: list[dict], prompts: list[dict]) -> dict:
    """
    Assign each word to the semantic chunk it falls into.
    Returns dict: chunk_index → list of word dicts.
    """
    chunk_words = {p["index"]: [] for p in prompts}

    for w in words:
        mid = (w["start"] + w["end"]) / 2
        for p in prompts:
            if p["start_sec"] <= mid < p["end_sec"]:
                chunk_words[p["index"]].append(w)
                break
        else:
            # Word falls outside all chunks — assign to nearest
            if prompts:
                closest = min(prompts, key=lambda p: abs(p["start_sec"] - mid))
                chunk_words[closest["index"]].append(w)

    return chunk_words


def build_pages(chunk_words: list[dict], font: ImageFont.FreeTypeFont) -> list[list[dict]]:
    """
    Split a chunk's words into pages that each fit within the text area
    and contain at most MAX_WORDS_PER_PAGE words.
    Pages break at line boundaries to keep lines intact.
    Returns list of pages, where each page is a list of word dicts.
    """
    text_width = VIDEO_WIDTH - 2 * TEXT_MARGIN_X
    ascent, descent = font.getmetrics()
    full_line_h = ascent + descent
    line_height = full_line_h + 16
    max_lines = max(1, (TEXT_AREA_HEIGHT - 40) // line_height)

    # First, wrap all words into lines using getlength
    lines = []
    current_line = []
    current_px = 0

    for w in chunk_words:
        word_text = w["word"] + " "
        word_advance = font.getlength(word_text)

        if current_px + word_advance > text_width and current_line:
            lines.append(current_line)
            current_line = [w]
            current_px = word_advance
        else:
            current_line.append(w)
            current_px += word_advance

    if current_line:
        lines.append(current_line)

    # Group lines into pages, respecting both max_lines and MAX_WORDS_PER_PAGE
    pages = []
    current_page_lines = []
    current_page_word_count = 0

    for line in lines:
        line_word_count = len(line)

        # Would adding this line exceed either limit?
        would_exceed_lines = len(current_page_lines) + 1 > max_lines
        would_exceed_words = current_page_word_count + line_word_count > MAX_WORDS_PER_PAGE

        if current_page_lines and (would_exceed_lines or would_exceed_words):
            # Flush current page
            pages.append([w for l in current_page_lines for w in l])
            current_page_lines = [line]
            current_page_word_count = line_word_count
        else:
            current_page_lines.append(line)
            current_page_word_count += line_word_count

    if current_page_lines:
        pages.append([w for l in current_page_lines for w in l])

    return pages


def get_current_page(pages: list[list[dict]], current_time: float) -> list[dict] | None:
    """
    Find which page should be displayed at the current time.
    A page stays visible from when its first word starts until its last word ends.
    """
    for page in pages:
        if not page:
            continue
        page_start = page[0]["start"]
        page_end = page[-1].get("end", page[-1]["start"] + 0.5)
        if page_start <= current_time < page_end:
            return page

    # If between pages or past the end, return None
    return None


def render_frame(bg_image: Image.Image, page_words: list[dict] | None,
                 current_time: float, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render a single video frame with background image and karaoke text.
    page_words is a stable set of words that all stay visible together.
    Text is center-aligned."""
    frame = bg_image.copy()
    draw = ImageDraw.Draw(frame, "RGBA")

    # Draw semi-transparent dark bar at bottom for text readability
    bar_top = VIDEO_HEIGHT - TEXT_AREA_HEIGHT
    draw.rectangle(
        [(0, bar_top), (VIDEO_WIDTH, VIDEO_HEIGHT)],
        fill=(0, 0, 0, BG_OVERLAY_ALPHA)
    )

    if not page_words:
        return np.array(frame.convert("RGB"))

    # Calculate available text width in pixels
    text_width = VIDEO_WIDTH - 2 * TEXT_MARGIN_X

    # Get consistent line height from font metrics
    ascent, descent = font.getmetrics()
    full_line_h = ascent + descent
    line_height = full_line_h + 16  # spacing between lines

    # Word-wrap the page into lines using getlength
    lines = []
    current_line_words = []
    current_line_px = 0

    for w in page_words:
        word_text = w["word"] + " "
        word_advance = font.getlength(word_text)

        if current_line_px + word_advance > text_width and current_line_words:
            lines.append(current_line_words)
            current_line_words = [w]
            current_line_px = word_advance
        else:
            current_line_words.append(w)
            current_line_px += word_advance

    if current_line_words:
        lines.append(current_line_words)

    # Vertically center the text block within the text area
    total_text_h = len(lines) * line_height
    y = bar_top + (TEXT_AREA_HEIGHT - total_text_h) // 2

    # Render each line — center-aligned, same font, highlight via background box
    for line_words in lines:
        # Measure line width: sum of all "word " advances, minus trailing space
        line_px = 0
        for i, w in enumerate(line_words):
            if i < len(line_words) - 1:
                line_px += font.getlength(w["word"] + " ")
            else:
                line_px += font.getlength(w["word"])

        # Center the line horizontally
        x = (VIDEO_WIDTH - line_px) / 2

        for i, w in enumerate(line_words):
            is_last = (i == len(line_words) - 1)
            word_text = w["word"] if is_last else w["word"] + " "
            word_only = w["word"]
            is_current = w["start"] <= current_time < w.get("end", w["start"] + 0.5)

            word_only_advance = font.getlength(word_only)
            word_full_advance = font.getlength(word_text)

            # Draw highlight background behind the current word
            if is_current:
                hx0 = x - HIGHLIGHT_PAD_X
                hy0 = y - HIGHLIGHT_PAD_Y
                hx1 = x + word_only_advance + HIGHLIGHT_PAD_X
                hy1 = y + full_line_h + HIGHLIGHT_PAD_Y
                draw.rounded_rectangle(
                    [(hx0, hy0), (hx1, hy1)],
                    radius=HIGHLIGHT_RADIUS,
                    fill=HIGHLIGHT_BG
                )

            # Shadow for readability
            draw.text((x + 2, y + 2), word_text, fill=SHADOW_COLOR, font=font)
            # Main text
            draw.text((x, y), word_text, fill=FONT_COLOR, font=font)

            x += word_full_advance

        y += line_height

    return np.array(frame.convert("RGB"))


# ─── Video Assembly ─────────────────────────────────────────────────────────

def assemble_video(audio_path: str, words: list[dict], prompts: list[dict],
                   image_map: dict, output_path: str,
                   bg_mode: str = "images", colors: list[tuple] = None):
    """
    Generate the video frame by frame and combine with audio using ffmpeg.
    bg_mode: "images" (static per chunk), "lava" (animated blobs), "waveform" (audio vis)
    """

    if colors is None:
        colors = parse_colors(",".join(DEFAULT_COLORS))

    # Get audio duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", audio_path],
        capture_output=True, text=True
    )
    total_duration = float(probe.stdout.strip())
    total_frames = int(total_duration * FPS)

    print(f"Audio duration: {total_duration:.1f}s")
    print(f"Total frames: {total_frames}")
    print(f"Semantic chunks: {len(prompts)}")
    print(f"Background mode: {bg_mode}")
    if bg_mode != "images":
        print(f"Colors: {['#%02x%02x%02x' % c for c in colors]}")
    else:
        print(f"Images found: {len(image_map)}")
    print()

    # Load font
    font = get_font(FONT_SIZE)

    # Precompute word pages per semantic chunk
    print("Assigning words to semantic chunks...")
    chunk_words = assign_words_to_chunks(words, prompts)
    chunk_pages = {}
    total_pages = 0
    for idx, cw in chunk_words.items():
        pages = build_pages(cw, font)
        chunk_pages[idx] = pages
        total_pages += len(pages)
        if cw:
            print(f"  Chunk {idx}: {len(cw)} words → {len(pages)} page(s)")

    print(f"  Total: {sum(len(cw) for cw in chunk_words.values())} words, {total_pages} pages")

    # Load background resources based on mode
    audio_samples = None
    audio_sr = None
    bg_cache = {}
    default_bg = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), (30, 30, 30))

    if bg_mode == "images":
        print("Loading background images...")
        for idx, img_path in image_map.items():
            try:
                bg_cache[idx] = load_and_resize_image(img_path)
                print(f"  Loaded image for chunk {idx}: {Path(img_path).name}")
            except Exception as e:
                print(f"  WARNING: Failed to load {img_path}: {e}")
    elif bg_mode == "waveform":
        print("Loading audio samples for spectrum analysis...")
        audio_samples, audio_sr = load_audio_samples(audio_path)
        print(f"  Loaded {len(audio_samples)} samples at {audio_sr} Hz")
    elif bg_mode == "radial":
        print("Loading audio samples for radial visualizer...")
        audio_samples, audio_sr = load_audio_samples(audio_path)
        print(f"  Loaded {len(audio_samples)} samples at {audio_sr} Hz")

    # Start ffmpeg pipe
    temp_video = output_path + ".temp.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",           # stdin
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        temp_video
    ]

    print(f"\nRendering video...")
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    current_chunk_idx = None
    current_bg = default_bg
    current_pages = []
    waveform_state = None  # for waveform mode (dict with spectrum + band_peaks)
    radial_state = None    # for radial mode (dict with spectrum + band_peaks)

    for frame_num in range(total_frames):
        current_time = frame_num / FPS

        # Determine which chunk we're in
        chunk_idx = get_chunk_for_time(prompts, current_time)

        # Update pages if chunk changed
        if chunk_idx != current_chunk_idx:
            current_chunk_idx = chunk_idx
            current_pages = chunk_pages.get(chunk_idx, [])
            # For images mode, update static background on chunk change
            if bg_mode == "images":
                if chunk_idx is not None and chunk_idx in bg_cache:
                    current_bg = bg_cache[chunk_idx]
                else:
                    current_bg = default_bg

        # Generate background based on mode
        if bg_mode == "lava":
            current_bg = generate_lava_bg(current_time, colors)
        elif bg_mode == "waveform":
            current_bg, waveform_state = generate_spectrum_bg(
                current_time, audio_samples, audio_sr, colors, waveform_state
            )
        elif bg_mode == "radial":
            current_bg, radial_state = generate_radial_bg(
                current_time, audio_samples, audio_sr, colors, radial_state
            )
        # For "images" mode, current_bg is already set above on chunk change

        # Find the current page within this chunk
        page = get_current_page(current_pages, current_time)

        # Render frame with the stable page of words
        frame = render_frame(current_bg, page, current_time, font)

        # Write to ffmpeg
        pipe.stdin.write(frame.tobytes())

        # Progress
        if frame_num % (FPS * 5) == 0:
            pct = (frame_num / total_frames) * 100
            print(f"  {current_time:.0f}s / {total_duration:.0f}s ({pct:.0f}%)")

    pipe.stdin.close()
    pipe.wait()

    if pipe.returncode != 0:
        stderr = pipe.stderr.read().decode()
        print(f"ERROR: ffmpeg failed:\n{stderr[-500:]}")
        sys.exit(1)

    # Mux audio
    print("\nAdding audio track...")
    final_cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    subprocess.run(final_cmd, check=True, capture_output=True)

    # Clean up temp
    Path(temp_video).unlink(missing_ok=True)

    print(f"\nDone! Video saved to: {output_path}")
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.1f} MB")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    global VIDEO_WIDTH, VIDEO_HEIGHT, FPS, FONT_SIZE
    parser = argparse.ArgumentParser(
        description="Assemble karaoke-style lecture video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # With generated images:
  python assemble_video.py --audio lecture.mp3 --words words.json \\
      --prompts prompts.jsonl --images output/images/ --output video.mp4

  # With lava lamp background:
  python assemble_video.py --audio lecture.mp3 --words words.json \\
      --prompts prompts.jsonl --bg-mode lava \\
      --colors "#1a1a2e,#16213e,#0f3460,#e94560" --output video.mp4

  # With waveform background:
  python assemble_video.py --audio lecture.mp3 --words words.json \\
      --prompts prompts.jsonl --bg-mode waveform \\
      --colors "#0d1b2a,#1b263b,#415a77,#778da9" --output video.mp4"""
    )
    parser.add_argument("--audio", required=True, help="Audio file (mp3/wav)")
    parser.add_argument("--words", required=True, help="Word timestamps JSON (from transcribe_words.py)")
    parser.add_argument("--prompts", required=True, help="Prompts JSONL (from generate_prompts.py)")
    parser.add_argument("--images", default=None, help="Directory with generated images (for --bg-mode images)")
    parser.add_argument("--output", required=True, help="Output video file (mp4)")
    parser.add_argument("--bg-mode", choices=["images", "lava", "waveform", "radial"], default="images",
                        help="Background mode (default: images)")
    parser.add_argument("--colors", default=None,
                        help='Comma-separated hex colors for lava/waveform mode '
                             '(e.g. "#1a1a2e,#16213e,#0f3460,#e94560")')
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH, help=f"Video width (default: {VIDEO_WIDTH})")
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT, help=f"Video height (default: {VIDEO_HEIGHT})")
    parser.add_argument("--fps", type=int, default=FPS, help=f"Frames per second (default: {FPS})")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help=f"Font size (default: {FONT_SIZE})")

    args = parser.parse_args()

    VIDEO_WIDTH = args.width
    VIDEO_HEIGHT = args.height
    FPS = args.fps
    FONT_SIZE = args.font_size

    bg_mode = args.bg_mode

    # Validate inputs
    for path, label in [(args.audio, "Audio"), (args.words, "Words JSON"),
                         (args.prompts, "Prompts JSONL")]:
        if not Path(path).exists():
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)

    if bg_mode == "images":
        if not args.images:
            print("ERROR: --images is required when using --bg-mode images")
            sys.exit(1)
        if not Path(args.images).is_dir():
            print(f"ERROR: Images directory not found: {args.images}")
            sys.exit(1)

    # Parse colors
    if args.colors:
        colors = parse_colors(args.colors)
    else:
        colors = parse_colors(",".join(DEFAULT_COLORS))

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: ffmpeg not found. Install it: sudo apt install ffmpeg")
        sys.exit(1)

    # Load data
    print("Loading data...")
    words = load_words(args.words)
    prompts = load_prompts(args.prompts)

    image_map = {}
    if bg_mode == "images":
        image_map = find_images(args.images, prompts)
        print(f"  Images: {len(image_map)} / {len(prompts)} chunks have images")
        missing = [p["index"] for p in prompts if p["index"] not in image_map]
        if missing:
            print(f"  WARNING: No images found for chunks: {missing}")
            print(f"  These chunks will show a black background.")

    print(f"  Words: {len(words)}")
    print(f"  Chunks: {len(prompts)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    assemble_video(args.audio, words, prompts, image_map, args.output,
                   bg_mode=bg_mode, colors=colors)


if __name__ == "__main__":
    main()
