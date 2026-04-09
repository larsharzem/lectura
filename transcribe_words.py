#!/usr/bin/env python3
"""
Transcribe audio with word-level timestamps using whisper-timestamped.
Outputs a JSON file with per-word timing data.

Usage:
    python transcribe_words.py audio.mp3 words.json
    python transcribe_words.py audio.mp3 words.json --model medium --language de

Install:
    pip install whisper-timestamped

This replaces your existing Whisper transcription step and provides the
word-level timing needed for karaoke-style video generation.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with word-level timestamps."
    )
    parser.add_argument("audio_file", help="Input audio file (mp3, wav, etc.)")
    parser.add_argument("output_json", help="Output JSON with word timestamps")
    parser.add_argument("--model", default="medium",
                        help="Whisper model size (tiny/base/small/medium/large) (default: medium)")
    parser.add_argument("--language", default=None,
                        help="Language code (e.g. 'de' for German). Auto-detected if omitted.")
    parser.add_argument("--device", default="cuda",
                        help="Device to run on (cuda/cpu) (default: cuda)")

    args = parser.parse_args()

    # Import here so the --help works without having whisper installed
    try:
        import whisper_timestamped as whisper
    except ImportError:
        print("ERROR: whisper-timestamped not installed.")
        print("Install it with: pip install whisper-timestamped")
        sys.exit(1)

    audio_path = Path(args.audio_file)
    output_path = Path(args.output_json)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Loading Whisper model '{args.model}' on {args.device}...")
    model = whisper.load_model(args.model, device=args.device)

    print(f"Transcribing {audio_path.name}...")
    kwargs = {}
    if args.language:
        kwargs["language"] = args.language

    audio = whisper.load_audio(str(audio_path))
    result = whisper.transcribe(model, audio, **kwargs)

    # Flatten to a simple word list for downstream use
    words = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info["text"].strip(),
                "start": round(word_info["start"], 3),
                "end": round(word_info["end"], 3),
                "confidence": round(word_info.get("confidence", 0.0), 3),
            })

    output = {
        "language": result.get("language", "unknown"),
        "audio_file": str(audio_path),
        "word_count": len(words),
        "words": words,
        # Also keep original segments for reference
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in result.get("segments", [])
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    duration = words[-1]["end"] if words else 0
    print(f"\nDone! {len(words)} words transcribed over {duration:.1f}s")
    print(f"Output: {output_path}")
    print(f"\nYou can now also use this for generate_prompts.py:")
    print(f"  The words.json contains segments that can be used for semantic chunking.")


if __name__ == "__main__":
    main()
