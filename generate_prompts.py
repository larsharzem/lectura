#!/usr/bin/env python3
"""
Splits a Whisper transcript into semantic chunks using a local LLM,
then generates metaphorical illustration prompts for each chunk.

Usage:
    python generate_prompts.py input/lecture.txt prompts/prompts.jsonl
    python generate_prompts.py input/lecture.txt prompts/prompts.jsonl --style tension
    python generate_prompts.py input/lecture.txt prompts/prompts.jsonl --style surreal
    python generate_prompts.py input/lecture.txt prompts/prompts.jsonl --style dialectic

    python generate_prompts.py --list-styles   # show all available styles

Requires: Ollama running locally with a model pulled (e.g. ollama pull llama3)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# --- Configuration ---
OLLAMA_MODEL = "llama3"       # or "mistral"
BLOCK_SECONDS = 150           # feed ~2.5 min blocks to the LLM for segmentation
WORDS_PER_SECOND = 2.5        # assumed speaking rate for plain text timing


# ─── Art Direction Styles ───────────────────────────────────────────────────
#
# Each style defines:
#   - system_prompt: the LLM personality / art direction instructions
#   - style_anchor: appended to every image prompt for SDXL consistency
#   - description: shown when listing styles
#

STYLES = {

    "editorial": {
        "description": (
            "Safe default. New Yorker-style editorial illustration. "
            "Clean metaphors, muted tones, tasteful. Can be generic."
        ),
        "system_prompt": """You are an art director at a serious magazine like The New Yorker or The Economist.
Your job: read a lecture excerpt and produce a SINGLE image generation prompt
that captures the core idea as a visual metaphor.

Rules:
- Think conceptual magazine illustration, book cover art
- Use clear visual metaphors (scales for justice, labyrinths for complexity, etc.)
- Muted, earthy, sophisticated color palette
- Avoid literal depictions (no podiums, classrooms, talking heads, computer screens)
- Avoid photorealism — painterly, textured, illustrated
- No text, labels, or words in the image
- One strong central image, not a busy scene""",
        "style_anchor": (
            "editorial illustration, conceptual metaphor, muted color palette, "
            "textured brushwork, symbolic imagery, sophisticated, "
            "no photorealism, no text, no labels"
        ),
    },

    "tension": {
        "description": (
            "Dramatic and confrontational. Emphasizes conflict, power, pressure. "
            "Strong diagonals, chiaroscuro, visual weight. Good for political content, "
            "debates about power, inequality, systemic critique."
        ),
        "system_prompt": """You are a politically engaged artist creating images for a radical publishing house.
Your job: read a lecture excerpt and produce a SINGLE image generation prompt
that makes the viewer FEEL the tension, contradiction, or stakes of the idea.

Rules:
- Every idea has a conflict — find it and make it visible
- Use dramatic composition: strong diagonals, crushing weight, precarious balance
- Chiaroscuro lighting — deep shadows against stark highlights
- Visual metaphors should be visceral, not cerebral: chains, cracks, floods, erosion,
  things buckling under pressure, hands grasping, structures tilting
- Scale contrasts: something enormous vs something fragile
- Dark palette with selective, almost violent color accents (deep red, burnt orange)
- No people speaking, no classrooms, no screens — only forces, structures, bodies under pressure
- Avoid photorealism — think Käthe Kollwitz, Francisco Goya, Otto Dix
- No text or labels""",
        "style_anchor": (
            "dramatic editorial illustration, heavy chiaroscuro, strong diagonals, "
            "dark moody palette with selective color accents, textured impasto, "
            "visceral symbolic imagery, expressionist influence, "
            "no photorealism, no text, no labels"
        ),
    },

    "surreal": {
        "description": (
            "Dreamlike and strange. Impossible spaces, scale distortions, "
            "melting boundaries. Good for philosophical content, epistemology, "
            "phenomenology, questions about perception and reality."
        ),
        "system_prompt": """You are a surrealist painter who has been asked to illustrate a philosophy lecture.
Your job: read a lecture excerpt and produce a SINGLE image generation prompt
that translates the idea into a dreamlike, impossible scene.

Rules:
- Treat every concept as a landscape or object that obeys dream logic, not physics
- Juxtapose things that don't belong together but create meaning through collision
- Distort scale: a keyhole the size of a cathedral, a thought the weight of a mountain
- Melt boundaries between categories: where does the body end and the city begin?
  Where does memory become architecture?
- Impossible spaces: Escher-like recursion, horizons that curve wrong,
  rooms that open into skies
- Quiet, eerie atmosphere — not chaotic, but uncanny. The viewer should feel
  slightly disoriented
- Palette: desaturated with isolated pools of unexpected color
- Think Giorgio de Chirico, Remedios Varo, René Magritte, Zdzisław Beksiński
- No text, no labels, no literal depictions of people lecturing""",
        "style_anchor": (
            "surrealist illustration, dreamlike impossible scene, "
            "uncanny atmosphere, distorted scale, desaturated palette with "
            "isolated color accents, painterly textured surface, "
            "metaphysical, no photorealism, no text, no labels"
        ),
    },

    "dialectic": {
        "description": (
            "Every image is a visual argument in two halves. Split compositions, "
            "diptych logic, before/after, thesis/antithesis. Good for content that "
            "compares, contrasts, or traces transformation."
        ),
        "system_prompt": """You are a visual essayist. Every image you create is a visual ARGUMENT with two sides.
Your job: read a lecture excerpt and produce a SINGLE image generation prompt
that structures the idea as a visual diptych or split composition.

Rules:
- Every image must have a clear duality: left vs right, top vs bottom,
  inside vs outside, surface vs depth
- The two halves should be in tension — they contradict, mirror, transform,
  or consume each other
- The boundary between the halves is where the meaning lives: a crack, a mirror,
  a horizon, a membrane, a hinge
- One side might be ordered and the other chaotic; one organic and one mechanical;
  one past and one future
- Use visual rhyming: shapes on one side echo distortedly on the other
- Muted, serious palette — think aged paper, iron, slate, amber
- This is intellectual illustration, not decoration. Every element argues.
- Think John Heartfield photomontage logic, but painted. Or Anselm Kiefer's material contrasts.
- No text, no labels, no literal scenes""",
        "style_anchor": (
            "diptych composition, split visual argument, contrasting halves, "
            "intellectual editorial illustration, muted serious palette, "
            "aged textures, painterly surface, conceptual, "
            "no photorealism, no text, no labels"
        ),
    },
}


# ─── Parsing ────────────────────────────────────────────────────────────────

def parse_srt(text: str) -> list[dict]:
    """Parse SRT format into list of {start_sec, end_sec, text}."""
    blocks = re.split(r'\n\n+', text.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        timecode = lines[1]
        match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            timecode
        )
        if not match:
            continue
        g = [int(x) for x in match.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        content = ' '.join(lines[2:]).strip()
        if content:
            entries.append({"start_sec": start, "end_sec": end, "text": content})
    return entries


def parse_plain_text(text: str) -> list[dict]:
    """For plain text (no timestamps), estimate timing from word count."""
    words = text.split()
    total_seconds = len(words) / WORDS_PER_SECOND
    return [{"start_sec": 0.0, "end_sec": round(total_seconds, 2), "text": text}]


def merge_into_blocks(entries: list[dict], block_sec: float) -> list[dict]:
    """Merge entries into larger blocks of ~block_sec seconds for segmentation."""
    if not entries:
        return []

    blocks = []
    current_texts = []
    block_start = entries[0]["start_sec"]

    for entry in entries:
        current_texts.append(entry["text"])
        elapsed = entry["end_sec"] - block_start

        if elapsed >= block_sec:
            blocks.append({
                "start_sec": round(block_start, 2),
                "end_sec": round(entry["end_sec"], 2),
                "text": ' '.join(current_texts)
            })
            current_texts = []
            block_start = entry["end_sec"]

    if current_texts:
        blocks.append({
            "start_sec": round(block_start, 2),
            "end_sec": round(entries[-1]["end_sec"], 2),
            "text": ' '.join(current_texts)
        })
    return blocks


# ─── Semantic Segmentation ──────────────────────────────────────────────────

def semantic_segment(text_block: str, start_sec: float, end_sec: float) -> list[dict]:
    """Ask the LLM to split a text block into semantic chunks."""
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=f"""You are analyzing a lecture transcript for semantic segmentation.

Split the following excerpt into distinct conceptual units — each time the
speaker moves to a new idea, argument, example, analogy, or topic shift,
that's a new segment. Segments can be anywhere from 1 sentence to several
paragraphs. Don't split mid-thought.

The excerpt covers {start_sec:.0f}s to {end_sec:.0f}s of audio.
Assume words are evenly distributed across this time range.

Respond ONLY with a JSON array. Each element must have:
- "text": the exact words belonging to this segment
- "summary": a 5-10 word summary of the idea

No other output. No markdown fences. No explanation. Just the JSON array.
The lecture may be in any language — always write the "summary" in English.

Transcript:
\"{text_block}\"""",
        capture_output=True, text=True, timeout=120
    )

    raw = result.stdout.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        segments = json.loads(raw)
    except json.JSONDecodeError:
        print(f"   WARNING: LLM returned invalid JSON for block {start_sec:.0f}s-{end_sec:.0f}s")
        print(f"   Raw output: {raw[:200]}...")
        print(f"   Falling back to single segment for this block.")
        return [{
            "text": text_block,
            "summary": "",
            "start_sec": start_sec,
            "end_sec": end_sec
        }]

    words = text_block.split()
    total_words = len(words)
    duration = end_sec - start_sec

    word_pos = 0
    for seg in segments:
        seg_words = len(seg.get("text", "").split())
        seg_start = start_sec + (word_pos / max(total_words, 1)) * duration
        seg_end = start_sec + ((word_pos + seg_words) / max(total_words, 1)) * duration
        seg["start_sec"] = round(seg_start, 2)
        seg["end_sec"] = round(seg_end, 2)
        word_pos += seg_words

    return segments


# ─── Prompt Generation ──────────────────────────────────────────────────────

def get_lecture_context(full_text: str) -> str:
    """Use LLM to summarize the lecture topic in one sentence."""
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=(
            "Summarize the following lecture in ONE sentence in English "
            "(just the topic/theme, no details). The lecture may be in any language. "
            "Respond with only the sentence.\n\n"
            f"{full_text[:3000]}"
        ),
        capture_output=True, text=True, timeout=60
    )
    return result.stdout.strip()


def generate_prompt(chunk_text: str, chunk_summary: str,
                    chunk_index: int, total_chunks: int,
                    lecture_context: str, style: dict) -> str:
    """Call Ollama to generate an illustration prompt from a semantic chunk."""
    summary_hint = f"\nThe core idea of this chunk: {chunk_summary}" if chunk_summary else ""

    system_msg = f"""{style['system_prompt']}

The lecture is about: {lecture_context}
This is chunk {chunk_index + 1} of {total_chunks}.{summary_hint}

The lecture may be in any language. Always respond in English.
Respond with ONLY the image prompt, nothing else. No explanation, no preamble, no "Here is...".
The prompt should be 1-3 sentences, under 200 words. Be specific and vivid — name
materials, lighting, composition. Do not be vague or generic."""

    user_msg = f"Lecture excerpt:\n\"{chunk_text}\""

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=f"System: {system_msg}\n\nUser: {user_msg}",
        capture_output=True, text=True, timeout=60
    )
    return result.stdout.strip()


# ─── Main ───────────────────────────────────────────────────────────────────

def list_styles():
    """Print all available styles with descriptions."""
    print("Available art direction styles:\n")
    for name, style in STYLES.items():
        print(f"  --style {name}")
        print(f"    {style['description']}\n")


def main():
    global OLLAMA_MODEL
    parser = argparse.ArgumentParser(
        description="Generate illustration prompts from a lecture transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python generate_prompts.py input/lecture.txt prompts/prompts.jsonl
  python generate_prompts.py input/lecture.srt prompts/prompts.jsonl --style tension
  python generate_prompts.py --list-styles"""
    )
    parser.add_argument("input_file", nargs="?", help="Whisper transcript (.txt or .srt)")
    parser.add_argument("output_file", nargs="?", help="Output path (e.g. prompts/prompts.jsonl)")
    parser.add_argument("--style", choices=STYLES.keys(), default="editorial",
                        help="Art direction style (default: editorial)")
    parser.add_argument("--model", default=OLLAMA_MODEL,
                        help=f"Ollama model to use (default: {OLLAMA_MODEL})")
    parser.add_argument("--list-styles", action="store_true",
                        help="List all available art direction styles and exit")
    parser.add_argument("--segments-only", action="store_true",
                        help="Only run semantic segmentation (pass 1), skip image prompt generation")

    args = parser.parse_args()

    if args.list_styles:
        list_styles()
        sys.exit(0)

    if not args.input_file or not args.output_file:
        parser.print_help()
        sys.exit(1)

    OLLAMA_MODEL = args.model

    style = STYLES[args.style]
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")

    # ── Parse input ──
    if input_path.suffix.lower() == ".srt":
        entries = parse_srt(text)
        blocks = merge_into_blocks(entries, BLOCK_SECONDS)
    else:
        entries = parse_plain_text(text)
        blocks = merge_into_blocks(entries, BLOCK_SECONDS)

    if not args.segments_only:
        print(f"Style: {args.style}")
        print(f"  {style['description']}\n")
    print(f"Parsed {input_path.name} → {len(blocks)} blocks for segmentation\n")

    # ── Get overall lecture context (only needed for image prompt generation) ──
    lecture_context = ""
    if not args.segments_only:
        full_text = ' '.join(b["text"] for b in blocks)
        lecture_context = get_lecture_context(full_text)
        print(f"Lecture topic: {lecture_context}\n")

    # ── Pass 1: Semantic segmentation ──
    print("═══ Pass 1: Semantic segmentation ═══\n")
    all_chunks = []
    for i, block in enumerate(blocks):
        print(f"Block {i+1}/{len(blocks)} ({block['start_sec']:.0f}s – {block['end_sec']:.0f}s) ...")
        segments = semantic_segment(block["text"], block["start_sec"], block["end_sec"])
        print(f"  → {len(segments)} semantic chunks found")
        for seg in segments:
            duration = seg["end_sec"] - seg["start_sec"]
            summary = seg.get("summary", "")
            print(f"    [{seg['start_sec']:.0f}s–{seg['end_sec']:.0f}s] ({duration:.0f}s) {summary}")
        all_chunks.extend(segments)

    print(f"\nTotal semantic chunks: {len(all_chunks)}")
    avg_duration = sum(c["end_sec"] - c["start_sec"] for c in all_chunks) / max(len(all_chunks), 1)
    print(f"Average chunk duration: {avg_duration:.1f}s\n")

    # ── Pass 2: Generate image prompts (skip if --segments-only) ──
    results = []
    if args.segments_only:
        print("═══ Skipping Pass 2 (--segments-only) ═══\n")
        for i, chunk in enumerate(all_chunks):
            results.append({
                "index": i,
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "duration_sec": round(chunk["end_sec"] - chunk["start_sec"], 2),
                "summary": chunk.get("summary", ""),
                "transcript": chunk["text"],
                "image_prompt": "",
                "style": "none"
            })
    else:
        print("═══ Pass 2: Generating image prompts ═══\n")
        for i, chunk in enumerate(all_chunks):
            summary = chunk.get("summary", "")
            label = summary if summary else chunk["text"][:60]
            print(f"[{i+1}/{len(all_chunks)}] {chunk['start_sec']:.0f}s–{chunk['end_sec']:.0f}s: {label} ...")

            prompt = generate_prompt(
                chunk["text"], summary, i, len(all_chunks), lecture_context, style
            )

            full_prompt = f"{prompt}, {style['style_anchor']}"

            results.append({
                "index": i,
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "duration_sec": round(chunk["end_sec"] - chunk["start_sec"], 2),
                "summary": summary,
                "transcript": chunk["text"],
                "image_prompt": full_prompt,
                "style": args.style
            })
            print(f"  → {prompt[:100]}...")

    # ── Write output ──
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_duration = all_chunks[-1]["end_sec"] - all_chunks[0]["start_sec"] if all_chunks else 0
    print(f"\n{'═' * 50}")
    print(f"Done! {len(results)} segments written to {output_path}")
    print(f"Lecture duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")
    print(f"Segments: {len(results)} (avg {avg_duration:.0f}s each)")
    if not args.segments_only:
        print(f"Style: {args.style}")
        print(f"\nNext steps:")
        print(f"  1. Review/edit {output_path}")
        print(f"  2. Run: python batch_generate.py {output_path} workflow_api.json")


if __name__ == "__main__":
    main()
