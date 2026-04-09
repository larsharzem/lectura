#!/usr/bin/env python3
"""
Illustrated lecture video pipeline: audio → AI-generated illustrations → video.

Steps:
  1. Transcribe audio with word-level timestamps
  2. Segment transcript + generate metaphorical image prompts (Ollama)
  3. Review/edit prompts
  4. Generate images via ComfyUI (SDXL + LoRA)
  5. Assemble karaoke video with images as backgrounds

Usage:
    python run_illustrated.py lecture.mp3 output/ --style tension --language de
    python run_illustrated.py --interactive
    python run_illustrated.py --list-styles

Settings are saved to config_illustrated.json and reused on future runs.

Prerequisites:
    pip install whisper-timestamped Pillow numpy
    sudo apt install ffmpeg
    ollama pull llama3
    ComfyUI running at http://127.0.0.1:8188
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from pipeline_common import (
    SCRIPT_DIR, WHISPER_MODELS,
    load_config, save_config, merge_args_over_config,
    ask, ask_int, ask_audio_and_output, ask_transcription, ask_video,
    banner, run_script, step_transcribe, ensure_transcript, step_assemble_video,
)


CONFIG_PATH = SCRIPT_DIR / "config_illustrated.json"

STYLES = ["editorial", "tension", "surreal", "dialectic"]

CONFIG_FIELDS = [
    "style", "language", "whisper_model", "ollama_model",
    "width", "height", "fps", "font_size", "comfyui_output", "workflow",
]

DEFAULTS = {
    "style": "editorial",
    "language": None,
    "whisper_model": "medium",
    "ollama_model": "llama3",
    "width": 1080,
    "height": 1080,
    "fps": 30,
    "font_size": 36,
    "comfyui_output": None,
    "workflow": None,
}

STEPS = {
    1: "Transcribe (word-level timestamps)",
    2: "Generate prompts (semantic chunking + art direction)",
    3: "Review prompts",
    4: "Generate images (ComfyUI)",
    5: "Assemble video",
}


# ─── Interactive ────────────────────────────────────────────────────────────

def interactive_setup(config: dict) -> dict:
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in config.items() if v is not None})

    print(f"\n{'═' * 60}")
    print(f"  Illustrated Video — Interactive Setup")
    print(f"{'═' * 60}")

    settings.update(ask_audio_and_output(settings))
    settings.update(ask_transcription(settings))

    print("\n── Art Direction ──")
    print("    editorial  — New Yorker-style clean metaphors")
    print("    tension    — dramatic, political, Kollwitz/Goya")
    print("    surreal    — dreamlike, de Chirico/Magritte")
    print("    dialectic  — split diptych, thesis vs antithesis")
    settings["style"] = ask("Art direction style",
                            default=settings.get("style", "editorial"),
                            options=STYLES)

    settings["ollama_model"] = ask("Ollama model for prompt generation",
                                   default=settings.get("ollama_model", "llama3"))

    settings.update(ask_video(settings))

    no_review = ask("Skip manual prompt review?", default="no", options=["yes", "no"])
    settings["no_review"] = (no_review == "yes")

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  Audio:      {settings['audio_file']}")
    print(f"  Output:     {settings['output_dir']}")
    print(f"  Language:   {settings.get('language') or 'auto-detect'}")
    print(f"  Style:      {settings['style']}")
    print(f"  Resolution: {settings['width']}×{settings['height']} @ {settings['fps']}fps")
    print()

    confirm = ask("Proceed?", default="yes", options=["yes", "no"])
    if confirm != "yes":
        print("  Aborted.")
        sys.exit(0)

    settings.setdefault("from_step", 1)
    settings.setdefault("skip_transcribe", False)
    settings.setdefault("skip_images", False)
    settings.setdefault("words", None)
    settings.setdefault("transcript", None)

    return settings


# ─── ComfyUI ───────────────────────────────────────────────────────────────

def check_comfyui():
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


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
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    video_path = output_dir / "lecture_video.mp4"
    transcript_path = (
        Path(s["transcript"]).resolve() if s.get("transcript")
        else output_dir / "transcript.txt"
    )
    workflow_path = (
        Path(s["workflow"]).resolve() if s.get("workflow")
        else SCRIPT_DIR / "workflow_api.json"
    )

    from_step = s.get("from_step", 1)
    if s.get("skip_transcribe") or s.get("words"):
        from_step = max(from_step, 2)

    # Plan
    print(f"\n{'═' * 60}")
    print(f"  Illustrated Video Pipeline")
    print(f"{'═' * 60}")
    print(f"  Audio:      {audio_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Style:      {s.get('style', 'editorial')}")
    print(f"  Language:   {s.get('language') or 'auto-detect'}")
    print()
    for num, desc in STEPS.items():
        skip = ""
        if num < from_step:
            skip = " [SKIP]"
        elif num == 3 and s.get("no_review"):
            skip = " [SKIP]"
        elif num == 4 and s.get("skip_images"):
            skip = " [SKIP]"
        print(f"  Step {num}: {desc}{skip}")
    print()

    # Step 1
    if from_step <= 1 and not s.get("skip_transcribe"):
        banner(1, 5, "Transcribe with word-level timestamps")
        step_transcribe(s, audio_path, words_path, transcript_path)
    else:
        ensure_transcript(s, words_path, transcript_path)

    # Step 2
    if from_step <= 2:
        banner(2, 5, f"Generate segments + prompts (style: {s.get('style', 'editorial')})")
        prompt_args = [
            str(transcript_path),
            str(prompts_path),
            "--style", s.get("style", "editorial"),
            "--model", s.get("ollama_model", "llama3"),
        ]
        run_script("generate_prompts.py", prompt_args)
    else:
        if not prompts_path.exists():
            print(f"ERROR: Prompts file not found: {prompts_path}")
            sys.exit(1)
        print(f"  Using existing prompts: {prompts_path}")

    # Step 3
    if from_step <= 3 and not s.get("no_review"):
        banner(3, 5, "Review prompts")
        with open(prompts_path) as f:
            num_prompts = sum(1 for line in f if line.strip())
        print(f"  {num_prompts} prompts generated in: {prompts_path}")
        print(f"\n  Edit the image_prompt fields, then press Enter to continue.")
        try:
            input("\n  Press Enter to continue... ")
        except KeyboardInterrupt:
            print(f"\n\n  Aborted. Resume with: python run_illustrated.py {audio_path} {output_dir} --from-step 4")
            sys.exit(0)

    # Step 4
    if from_step <= 4 and not s.get("skip_images"):
        banner(4, 5, "Generate images via ComfyUI")

        if not workflow_path.exists():
            print(f"  ERROR: Workflow not found: {workflow_path}")
            sys.exit(1)

        if not check_comfyui():
            print(f"  ERROR: ComfyUI is not running at http://127.0.0.1:8188")
            print(f"\n  Resume with: python run_illustrated.py {audio_path} {output_dir} --from-step 4")
            sys.exit(1)

        comfyui_output = (
            Path(s["comfyui_output"]) if s.get("comfyui_output")
            else Path.home() / "ComfyUI" / "output"
        )

        # Clear old images
        if comfyui_output.exists():
            with open(prompts_path) as f:
                prompt_indices = [json.loads(line)["index"] for line in f if line.strip()]
            cleared = 0
            for idx in prompt_indices:
                prefix = f"{idx:04d}"
                for old_img in comfyui_output.glob(f"{prefix}*"):
                    if old_img.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                        old_img.unlink()
                        cleared += 1
            if cleared:
                print(f"  Cleared {cleared} old images from {comfyui_output}")

        run_script("batch_generate.py", [str(prompts_path), str(workflow_path)])

        # Copy images
        if comfyui_output.exists():
            print(f"\n  Copying images from {comfyui_output} → {images_dir}")
            copied = 0
            for idx in prompt_indices:
                prefix = f"{idx:04d}"
                candidates = sorted(
                    [f for f in comfyui_output.glob(f"{prefix}*")
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")],
                    key=lambda f: f.stat().st_mtime, reverse=True
                )
                if candidates:
                    shutil.copy2(candidates[0], images_dir / f"{prefix}.png")
                    copied += 1
            print(f"  Copied {copied} images")
    else:
        print(f"  Using existing images in: {images_dir}")

    # Step 5
    banner(5, 5, "Assemble karaoke video")
    step_assemble_video(s, audio_path, words_path, prompts_path,
                        video_path, bg_mode="images", images_dir=images_dir)

    # Done
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete!")
    print(f"{'═' * 60}")
    print(f"  Video:   {video_path}")
    print(f"  Images:  {images_dir}")
    print(f"  Config:  {CONFIG_PATH}")
    print(f"\n  Re-run with different style:")
    print(f"    python run_illustrated.py {audio_path} {output_dir} --style surreal --from-step 2")
    print()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Illustrated lecture video: audio → AI images → karaoke video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_illustrated.py lecture.mp3 output/ --style tension --language de
  python run_illustrated.py --interactive
  python run_illustrated.py --list-styles"""
    )

    parser.add_argument("audio_file", nargs="?", help="Input audio file")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive setup wizard")
    parser.add_argument("--list-styles", action="store_true",
                        help="List available styles and exit")

    parser.add_argument("--style", default="editorial", help="Art direction style")
    parser.add_argument("--language", default=None, help="Audio language code")
    parser.add_argument("--whisper-model", default="medium", dest="whisper_model")
    parser.add_argument("--ollama-model", default="llama3", dest="ollama_model")
    parser.add_argument("--workflow", default=None, help="Path to workflow_api.json")
    parser.add_argument("--comfyui-output", default=None, dest="comfyui_output")

    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--font-size", type=int, default=36, dest="font_size")

    parser.add_argument("--from-step", type=int, default=1, choices=range(1, 6),
                        dest="from_step")
    parser.add_argument("--skip-transcribe", action="store_true", dest="skip_transcribe")
    parser.add_argument("--words", default=None)
    parser.add_argument("--skip-images", action="store_true", dest="skip_images")
    parser.add_argument("--no-review", action="store_true", dest="no_review")
    parser.add_argument("--transcript", default=None)

    args = parser.parse_args()

    if args.list_styles:
        run_script("generate_prompts.py", ["--list-styles"], check=False)
        sys.exit(0)

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
