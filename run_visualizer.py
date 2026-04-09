#!/usr/bin/env python3
"""
Visualizer lecture video pipeline: audio → animated background → video.

Steps:
  1. Transcribe audio with word-level timestamps
  2. Segment transcript into semantic chunks
  3. Assemble karaoke video with animated background

Background modes:
  lava     — animated abstract blobs
  waveform — reactive FFT frequency spectrum bars
  radial   — circular spectrum ring radiating from center

Usage:
    python run_visualizer.py lecture.mp3 output/ --mode radial --colors "#0a0a0a,#e94560,#0f3460"
    python run_visualizer.py lecture.mp3 output/ --mode waveform
    python run_visualizer.py --interactive

Settings are saved to config_visualizer.json and reused on future runs.

Prerequisites:
    pip install whisper-timestamped Pillow numpy
    sudo apt install ffmpeg
    ollama pull llama3  (for semantic segmentation)
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline_common import (
    SCRIPT_DIR, WHISPER_MODELS,
    load_config, save_config, merge_args_over_config,
    ask, ask_int, ask_audio_and_output, ask_transcription, ask_video,
    banner, run_script, step_transcribe, ensure_transcript, step_assemble_video,
)


CONFIG_PATH = SCRIPT_DIR / "config_visualizer.json"

BG_MODES = ["lava", "waveform", "radial"]

CONFIG_FIELDS = [
    "mode", "colors", "language", "whisper_model", "ollama_model",
    "width", "height", "fps", "font_size",
]

DEFAULTS = {
    "mode": "radial",
    "colors": "#1a1a2e,#16213e,#0f3460,#e94560",
    "language": None,
    "whisper_model": "medium",
    "ollama_model": "llama3",
    "width": 1080,
    "height": 1080,
    "fps": 30,
    "font_size": 36,
}

STEPS = {
    1: "Transcribe (word-level timestamps)",
    2: "Semantic segmentation",
    3: "Assemble video",
}


# ─── Interactive ────────────────────────────────────────────────────────────

def interactive_setup(config: dict) -> dict:
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in config.items() if v is not None})

    print(f"\n{'═' * 60}")
    print(f"  Visualizer Video — Interactive Setup")
    print(f"{'═' * 60}")

    settings.update(ask_audio_and_output(settings))
    settings.update(ask_transcription(settings))

    print("\n── Visualization ──")
    print("    lava     — animated abstract blobs")
    print("    waveform — reactive frequency spectrum bars")
    print("    radial   — circular spectrum ring")
    settings["mode"] = ask("Visualization mode",
                           default=settings.get("mode", "radial"),
                           options=BG_MODES)

    print("\n── Colors ──")
    print("    Comma-separated hex colors. First = background, rest = visualization.")
    print(f"    Example: #0d1b2a,#1b263b,#415a77,#778da9")
    settings["colors"] = ask("Color palette",
                             default=settings.get("colors", "#1a1a2e,#16213e,#0f3460,#e94560"))

    settings.update(ask_video(settings))

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  Audio:      {settings['audio_file']}")
    print(f"  Output:     {settings['output_dir']}")
    print(f"  Language:   {settings.get('language') or 'auto-detect'}")
    print(f"  Mode:       {settings['mode']}")
    print(f"  Colors:     {settings['colors']}")
    print(f"  Resolution: {settings['width']}×{settings['height']} @ {settings['fps']}fps")
    print()

    confirm = ask("Proceed?", default="yes", options=["yes", "no"])
    if confirm != "yes":
        print("  Aborted.")
        sys.exit(0)

    settings.setdefault("from_step", 1)
    settings.setdefault("skip_transcribe", False)
    settings.setdefault("words", None)
    settings.setdefault("transcript", None)

    return settings


# ─── Pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(s: dict):
    audio_path = Path(s["audio_file"]).resolve()
    output_dir = Path(s["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    words_path = Path(s["words"]).resolve() if s.get("words") else output_dir / "words.json"
    prompts_path = output_dir / "prompts.jsonl"
    video_path = output_dir / "lecture_video.mp4"
    transcript_path = (
        Path(s["transcript"]).resolve() if s.get("transcript")
        else output_dir / "transcript.txt"
    )

    mode = s.get("mode", "radial")

    from_step = s.get("from_step", 1)
    if s.get("skip_transcribe") or s.get("words"):
        from_step = max(from_step, 2)

    # Plan
    print(f"\n{'═' * 60}")
    print(f"  Visualizer Video Pipeline")
    print(f"{'═' * 60}")
    print(f"  Audio:      {audio_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Mode:       {mode}")
    print(f"  Colors:     {s.get('colors', '')}")
    print(f"  Language:   {s.get('language') or 'auto-detect'}")
    print()
    for num, desc in STEPS.items():
        skip = " [SKIP]" if num < from_step else ""
        print(f"  Step {num}: {desc}{skip}")
    print()

    # Step 1
    if from_step <= 1 and not s.get("skip_transcribe"):
        banner(1, 3, "Transcribe with word-level timestamps")
        step_transcribe(s, audio_path, words_path, transcript_path)
    else:
        ensure_transcript(s, words_path, transcript_path)

    # Step 2
    if from_step <= 2:
        banner(2, 3, "Semantic segmentation")
        prompt_args = [
            str(transcript_path),
            str(prompts_path),
            "--segments-only",
            "--model", s.get("ollama_model", "llama3"),
        ]
        run_script("generate_prompts.py", prompt_args)
    else:
        if not prompts_path.exists():
            print(f"ERROR: Prompts file not found: {prompts_path}")
            sys.exit(1)
        print(f"  Using existing segments: {prompts_path}")

    # Step 3
    banner(3, 3, f"Assemble video ({mode})")
    step_assemble_video(s, audio_path, words_path, prompts_path,
                        video_path, bg_mode=mode)

    # Done
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete!")
    print(f"{'═' * 60}")
    print(f"  Video:   {video_path}")
    print(f"  Config:  {CONFIG_PATH}")
    print(f"\n  Re-run with different mode:")
    print(f"    python run_visualizer.py {audio_path} {output_dir} --mode waveform --from-step 3")
    print(f"\n  Re-run with different colors:")
    print(f"    python run_visualizer.py {audio_path} {output_dir} --colors '#...' --from-step 3")
    print()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualizer lecture video: audio → animated background → karaoke video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_visualizer.py lecture.mp3 output/ --mode radial --colors "#0a0a0a,#e94560,#0f3460"
  python run_visualizer.py lecture.mp3 output/ --mode waveform
  python run_visualizer.py --interactive"""
    )

    parser.add_argument("audio_file", nargs="?", help="Input audio file")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive setup wizard")

    parser.add_argument("--mode", default="radial", choices=BG_MODES,
                        help="Visualization mode (default: radial)")
    parser.add_argument("--colors", default="#1a1a2e,#16213e,#0f3460,#e94560",
                        help="Comma-separated hex colors")
    parser.add_argument("--language", default=None, help="Audio language code")
    parser.add_argument("--whisper-model", default="medium", dest="whisper_model")
    parser.add_argument("--ollama-model", default="llama3", dest="ollama_model")

    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--font-size", type=int, default=36, dest="font_size")

    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 4),
                        dest="from_step", help="Resume from step (1-3)")
    parser.add_argument("--skip-transcribe", action="store_true", dest="skip_transcribe")
    parser.add_argument("--words", default=None)
    parser.add_argument("--transcript", default=None)

    args = parser.parse_args()

    config = load_config(CONFIG_PATH)

    if args.interactive:
        settings = interactive_setup(config)
    else:
        settings = merge_args_over_config(args, config, parser, CONFIG_FIELDS)
        if not settings.get("audio_file") or not settings.get("output_dir"):
            parser.print_help()
            sys.exit(1)

    save_config(CONFIG_PATH, settings, CONFIG_FIELDS)
    print(f"  Settings saved to {CONFIG_PATH}")

    run_pipeline(settings)


if __name__ == "__main__":
    main()
