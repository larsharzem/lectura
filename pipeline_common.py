"""
Shared utilities for the lecture-illustrator pipeline.
Used by run_illustrated.py and run_visualizer.py.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


# ─── Config ─────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load saved config, falling back to empty dict."""
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def save_config(config_path: Path, settings: dict, fields: list[str]):
    """Save specified fields from settings to config JSON."""
    to_save = {}
    for k in fields:
        if k in settings and settings[k] is not None:
            to_save[k] = settings[k]
    config_path.write_text(json.dumps(to_save, indent=2) + "\n")


def merge_args_over_config(args: argparse.Namespace, config: dict,
                           parser: argparse.ArgumentParser,
                           config_fields: list[str]) -> dict:
    """
    Merge CLI arguments over config. CLI flags win over saved config.
    Only override config if the user explicitly passed the flag.
    """
    explicit = set()
    for action in parser._actions:
        if action.dest in ("help", "interactive"):
            continue
        val = getattr(args, action.dest, None)
        if val != action.default:
            explicit.add(action.dest)

    settings = dict(config)
    for k in config_fields:
        if k in explicit:
            settings[k] = getattr(args, k)
        elif k not in settings:
            # Use argparse default if not in config
            if hasattr(args, k):
                settings[k] = getattr(args, k)

    # Non-config args always come from CLI
    for attr in dir(args):
        if not attr.startswith("_") and attr not in config_fields:
            settings[attr] = getattr(args, attr)

    return settings


# ─── Interactive Helpers ────────────────────────────────────────────────────

def ask(prompt: str, default=None, options: list = None) -> str:
    """Ask the user for input with a default value shown."""
    if options:
        options_str = ", ".join(options)
        prompt_full = f"  {prompt} [{options_str}]"
    else:
        prompt_full = f"  {prompt}"

    if default is not None and default != "":
        prompt_full += f" (default: {default})"

    prompt_full += ": "

    while True:
        answer = input(prompt_full).strip()
        if not answer:
            return default

        if options and answer not in options:
            print(f"    Please choose from: {', '.join(options)}")
            continue

        return answer


def ask_int(prompt: str, default: int) -> int:
    """Ask for an integer with a default."""
    while True:
        answer = input(f"  {prompt} (default: {default}): ").strip()
        if not answer:
            return default
        try:
            return int(answer)
        except ValueError:
            print("    Please enter a number.")


def ask_audio_and_output(config: dict) -> dict:
    """Interactive: ask for audio file and output directory."""
    settings = dict(config)

    print("\n── Audio ──")
    audio = ask("Audio file path", default=settings.get("audio_file", ""))
    if not audio:
        print("\n  ERROR: Audio file is required.")
        sys.exit(1)
    settings["audio_file"] = audio

    output = ask("Output directory", default=settings.get("output_dir", "output/"))
    settings["output_dir"] = output

    return settings


def ask_transcription(config: dict) -> dict:
    """Interactive: ask for language and whisper model."""
    settings = dict(config)

    print("\n── Transcription ──")
    lang = ask("Audio language code (e.g. 'de', 'en', 'fr', or empty for auto-detect)",
               default=settings.get("language") or "")
    settings["language"] = lang if lang else None

    wm = ask("Whisper model size", default=settings.get("whisper_model", "medium"),
             options=WHISPER_MODELS)
    settings["whisper_model"] = wm

    return settings


def ask_video(config: dict) -> dict:
    """Interactive: ask for video dimensions."""
    settings = dict(config)

    print("\n── Video ──")
    settings["width"] = ask_int("Video width (px)", default=settings.get("width", 1080))
    settings["height"] = ask_int("Video height (px)", default=settings.get("height", 1080))
    settings["fps"] = ask_int("Frames per second", default=settings.get("fps", 30))
    settings["font_size"] = ask_int("Subtitle font size", default=settings.get("font_size", 36))

    return settings


# ─── Pipeline Utilities ────────────────────────────────────────────────────

def banner(step_num: int, total: int, text: str):
    print(f"\n{'═' * 60}")
    print(f"  Step {step_num}/{total}: {text}")
    print(f"{'═' * 60}\n")


def run_script(script: str, args: list[str], check: bool = True):
    """Run a sibling Python script with arguments."""
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + args
    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\n  ERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)
    return result


# ─── Common Steps ───────────────────────────────────────────────────────────

def step_transcribe(s: dict, audio_path: Path, words_path: Path,
                    transcript_path: Path):
    """Step: Transcribe audio with word-level timestamps."""
    transcribe_args = [
        str(audio_path),
        str(words_path),
        "--model", s.get("whisper_model", "medium"),
        "--device", "cuda",
    ]
    if s.get("language"):
        transcribe_args += ["--language", s["language"]]

    run_script("transcribe_words.py", transcribe_args)

    # Generate plain transcript for segmentation
    if not s.get("transcript"):
        with open(words_path) as f:
            words_data = json.load(f)
        full_text = " ".join(w["word"] for w in words_data["words"])
        transcript_path.write_text(full_text, encoding="utf-8")
        print(f"  Plain transcript saved to: {transcript_path}")


def ensure_transcript(s: dict, words_path: Path, transcript_path: Path):
    """Ensure words.json and transcript.txt exist when skipping transcription."""
    if not words_path.exists():
        print(f"ERROR: Words file not found: {words_path}")
        print("  Run transcription first, or provide --words <path>")
        sys.exit(1)
    print(f"  Using existing words: {words_path}")

    if not s.get("transcript") and not transcript_path.exists():
        with open(words_path) as f:
            words_data = json.load(f)
        full_text = " ".join(w["word"] for w in words_data["words"])
        transcript_path.write_text(full_text, encoding="utf-8")


def step_assemble_video(s: dict, audio_path: Path, words_path: Path,
                        prompts_path: Path, video_path: Path,
                        bg_mode: str, images_dir: Path = None):
    """Step: Assemble the final video."""
    video_args = [
        "--audio", str(audio_path),
        "--words", str(words_path),
        "--prompts", str(prompts_path),
        "--output", str(video_path),
        "--bg-mode", bg_mode,
        "--width", str(s.get("width", 1080)),
        "--height", str(s.get("height", 1080)),
        "--fps", str(s.get("fps", 30)),
        "--font-size", str(s.get("font_size", 36)),
    ]
    if bg_mode == "images" and images_dir:
        video_args += ["--images", str(images_dir)]
    if s.get("colors"):
        video_args += ["--colors", s["colors"]]

    run_script("assemble_video.py", video_args)
