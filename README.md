# Lectura

Turn lecture audio into illustrated karaoke-style videos — entirely locally.

Given an audio file of a lecture, this pipeline:

1. **Transcribes** the audio with per-word timestamps
2. **Segments** the transcript into semantic chunks (by idea, not by time)
3. **Generates metaphorical illustration prompts** for each chunk using a local LLM
4. **Generates images** from those prompts using Stable Diffusion (SDXL) via ComfyUI
5. **Assembles a video** with the images as backgrounds and karaoke-style word highlighting

Alternatively, skip image generation and use animated backgrounds:
- **Lava lamp** — abstract evolving blobs
- **Waveform** — reactive FFT frequency spectrum bars
- **Radial** — circular spectrum ring radiating from center

Everything runs locally. No cloud APIs, no subscriptions.

---

## Demo

## Quick Start

```bash
# AI-generated illustrations
python run_illustrated.py lecture.mp3 output/ --style tension --language de

# Animated spectrum background (no ComfyUI needed)
python run_visualizer.py lecture.mp3 output/ --mode waveform \
    --colors "#0d1b2a,#1b263b,#415a77,#778da9" --language de

# Circular spectrum ring
python run_visualizer.py lecture.mp3 output/ --mode radial \
    --colors "#0a0a0a,#e94560,#0f3460,#16213e" --language de

# Interactive setup (walks you through all options)
python run_illustrated.py --interactive
python run_visualizer.py --interactive
```

---

## Requirements

### Hardware

- **GPU**: NVIDIA with 8+ GB VRAM (tested on RTX 5060 Ti 16GB)
  - 8 GB is sufficient for SDXL image generation
  - Whisper transcription also benefits from GPU acceleration
- **RAM**: 16 GB minimum, 32 GB recommended
- **Disk**: ~25 GB for models (Whisper + SDXL checkpoint + LoRA)

### Software

- **OS**: Linux or WSL2 on Windows 11
- **Python**: 3.11+
- **NVIDIA drivers**: Latest for your GPU
- **ffmpeg**: For audio processing and video encoding

---

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/larsharzem/lectura.git
cd lectura
```

### 2. Create a Python environment

```bash
# Using conda (recommended)
conda create --name lectureillustrator python=3.11 -y
conda activate lectureillustrator

# Or using venv
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

Check your GPU architecture. For RTX 40-series and older:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For RTX 50-series (Blackwell / sm_120), you need nightly builds:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

### 4. Install Python dependencies

```bash
pip install whisper-timestamped Pillow numpy
```

### 5. Install ffmpeg

```bash
# Ubuntu / WSL
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

### 6. Install Ollama (for LLM-based prompt generation)

Only needed for `--bg-mode images` (AI-generated illustrations).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3    # or: ollama pull mistral
```

### 7. Set up ComfyUI (for AI image generation)

Only needed for `--bg-mode images`.

```bash
# Clone ComfyUI
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# Install ComfyUI Manager (for easy model/node management)
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ..
```

#### Download models

**SDXL Checkpoint** — place in `~/ComfyUI/models/checkpoints/`:
- Go to [civitai.com](https://civitai.com), filter by Base Model → SDXL 1.0, Type → Checkpoint
- Recommended: DreamShaper XL, Juggernaut XL

**Illustration LoRA** (optional, for style control) — place in `~/ComfyUI/models/loras/`:
- Filter Civitai by: Type → LoRA, Base Model → SDXL 1.0
- Recommended: ClassipeintXL (oil painting), or search for illustration/editorial styles
- Always check sample images before downloading

#### Create and export a workflow

1. Start ComfyUI: `python main.py`
2. Open `http://127.0.0.1:8188` in your browser
3. Build a workflow with these nodes:
   - **Load Checkpoint** → select your SDXL checkpoint
   - **Load LoRA (Model and CLIP)** → select your LoRA (optional)
   - **CLIP Text Encode (Prompt)** × 2 → one for positive, one for negative
   - **EmptyLatentImage** → set width and height to 1024
   - **KSampler**
   - **VAE Decode**
   - **SaveImage**
4. Wire them:
   ```
   Checkpoint → [LoRA] → CLIP Text Encode (pos) → KSampler (positive)
                        → CLIP Text Encode (neg) → KSampler (negative)
   Checkpoint → VAE                              → VAE Decode (vae)
   EmptyLatentImage                              → KSampler (latent_image)
   KSampler                                      → VAE Decode (samples)
   VAE Decode                                    → SaveImage
   ```
5. Settings (gear icon) → enable **Dev mode options**
6. Click **Save (API Format)** → save as `workflow_api.json`
7. Copy it into the repo directory:
   ```bash
   cp workflow_api.json ~/lectura/
   ```

The `batch_generate.py` script auto-detects node IDs from your workflow, so any valid ComfyUI workflow with the standard node types will work.

### 8. Font (optional)

The pipeline uses system fonts by default. For a nicer look, place a TTF font in the repo directory:

```bash
# Example: download Poppins
wget -O Poppins-Regular.ttf "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf"
```

The script checks `./Poppins-Regular.ttf` first, then falls back to system fonts (DejaVu Sans, Liberation Sans, etc.).

---

## Usage

There are two pipelines, each with its own script:

| Script | What it does | Needs ComfyUI? | Needs Ollama? |
|--------|-------------|----------------|---------------|
| `run_illustrated.py` | AI-generated illustration backgrounds | Yes | Yes |
| `run_visualizer.py` | Animated audio visualization backgrounds | No | Yes (for segmentation) |

Both share transcription (Whisper) and video assembly. Settings are saved to separate config files (`config_illustrated.json` / `config_visualizer.json`).

### Interactive mode

The easiest way to get started. Walks you through every option with sensible defaults:

```bash
python run_illustrated.py --interactive
python run_visualizer.py --interactive
```

### Illustrated pipeline

Generates metaphorical images from the lecture content using a local LLM + Stable Diffusion.

```bash
python run_illustrated.py lecture.mp3 output/ --style tension --language de
```

| Flag | Default | Description |
|------|---------|-------------|
| `--style` | `editorial` | Art direction: `editorial`, `tension`, `surreal`, `dialectic` |
| `--language` | auto | Audio language code (e.g. `de`, `en`, `fr`) |
| `--whisper-model` | `medium` | Whisper model size |
| `--ollama-model` | `llama3` | LLM for prompt generation |
| `--workflow` | `./workflow_api.json` | ComfyUI workflow file |
| `--comfyui-output` | `~/ComfyUI/output` | ComfyUI image output directory |
| `--width` | `1080` | Video width |
| `--height` | `1080` | Video height |
| `--fps` | `30` | Frames per second |
| `--font-size` | `36` | Subtitle font size |
| `--from-step N` | `1` | Resume from step (1–5) |
| `--skip-transcribe` | — | Skip transcription |
| `--skip-images` | — | Reuse existing images |
| `--no-review` | — | Skip prompt review pause |
| `-i`, `--interactive` | — | Interactive setup wizard |

### Visualizer pipeline

Animated backgrounds that react to the audio. No image generation needed.

```bash
# Frequency spectrum bars
python run_visualizer.py lecture.mp3 output/ --mode waveform \
    --colors "#0d1b2a,#1b263b,#415a77,#778da9"

# Circular spectrum ring
python run_visualizer.py lecture.mp3 output/ --mode radial \
    --colors "#0a0a0a,#e94560,#0f3460,#16213e"

# Lava lamp blobs
python run_visualizer.py lecture.mp3 output/ --mode lava \
    --colors "#1a1a2e,#16213e,#0f3460,#e94560"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `radial` | Visualization: `lava`, `waveform`, `radial` |
| `--colors` | dark palette | Comma-separated hex colors (first = background) |
| `--language` | auto | Audio language code |
| `--whisper-model` | `medium` | Whisper model size |
| `--ollama-model` | `llama3` | LLM for segmentation |
| `--width` | `1080` | Video width |
| `--height` | `1080` | Video height |
| `--fps` | `30` | Frames per second |
| `--font-size` | `36` | Subtitle font size |
| `--from-step N` | `1` | Resume from step (1–3) |
| `--skip-transcribe` | — | Skip transcription |
| `-i`, `--interactive` | — | Interactive setup wizard |

### Config files

Settings are saved automatically after every run. On the next run, saved values are used as defaults — you only need to specify what changed:

```bash
# First run: specify everything
python run_visualizer.py lecture.mp3 output/ --mode radial \
    --colors "#0a0a0a,#e94560,#0f3460" --language de

# Second run: mode, colors, language remembered
python run_visualizer.py lecture2.mp3 output2/

# Override just one setting
python run_visualizer.py lecture3.mp3 output3/ --mode waveform
```

CLI flags always take priority over saved config.
| `--from-step N` | `1` | Resume from step N (1–5) |
| `--skip-transcribe` | — | Skip transcription, requires `--words` |
| `--words` | — | Path to existing `words.json` |
| `--skip-images` | — | Skip image generation, reuse existing |
| `--no-review` | — | Skip the manual prompt review pause |

### Art direction styles

```bash
python run_pipeline.py --list-styles
```

| Style | Best for | Visual approach |
|-------|----------|-----------------|
| `editorial` | General | New Yorker-style clean metaphors, muted tones |
| `tension` | Political content | Dramatic chiaroscuro, conflict, pressure, Kollwitz/Goya |
| `surreal` | Philosophy | Dreamlike impossible spaces, de Chirico/Magritte |
| `dialectic` | Comparisons | Split diptych compositions, thesis vs antithesis |

### Color palettes for animated backgrounds

The `--colors` flag accepts comma-separated hex colors:

- **First color** = background
- **Remaining colors** = visualization elements (bars, blobs, ring)

For `radial` mode, the inner circle is always a neutral outline — all palette colors go to the bars.

Examples:

```bash
# Dark blue to red (good for speech/lecture)
--colors "#0d1b2a,#1b263b,#415a77,#778da9,#e0e1dd"

# Warm sunset
--colors "#0a0a0a,#d4a574,#e8956a,#c1666b,#4a4a4a"

# Neon
--colors "#0a0a0a,#00f0ff,#7b2ff7,#ff2d95,#ffcc00"

# Monochrome
--colors "#0a0a0a,#333333,#666666,#999999,#cccccc"
```

---

## Running individual scripts

Each script can be run independently:

### Transcribe with word-level timestamps

```bash
python transcribe_words.py audio.mp3 words.json --language de --model medium
```

### Generate semantic segments + image prompts

```bash
python generate_prompts.py transcript.txt prompts.jsonl --style tension

# Segments only (no image prompts, for animated background modes)
python generate_prompts.py transcript.txt prompts.jsonl --segments-only
```

### Batch generate images via ComfyUI

```bash
# ComfyUI must be running
python batch_generate.py prompts.jsonl workflow_api.json
```

### Assemble video

```bash
# With images
python assemble_video.py --audio lecture.mp3 --words words.json \
    --prompts prompts.jsonl --images output/images/ --output video.mp4

# With animated background
python assemble_video.py --audio lecture.mp3 --words words.json \
    --prompts prompts.jsonl --bg-mode radial \
    --colors "#0a0a0a,#e94560,#0f3460,#16213e" --output video.mp4
```

---

## Resuming / re-running

Both pipelines save intermediate artifacts, so you can re-run individual steps:

```bash
# Illustrated: re-run with a different art style (skip transcription)
python run_illustrated.py lecture.mp3 output/ --style surreal --from-step 2

# Illustrated: re-generate images only (keep same prompts)
python run_illustrated.py lecture.mp3 output/ --from-step 4

# Illustrated: rebuild video only (after editing images)
python run_illustrated.py lecture.mp3 output/ --from-step 5

# Visualizer: switch mode without re-transcribing
python run_visualizer.py lecture.mp3 output/ --mode waveform --from-step 3

# Visualizer: change colors only
python run_visualizer.py lecture.mp3 output/ --colors "#..." --from-step 3
```

---

## Output structure

```
project/
├── config_illustrated.json    # Saved settings for illustrated pipeline
├── config_visualizer.json     # Saved settings for visualizer pipeline
├── workflow_api.json          # Your ComfyUI workflow (illustrated only)
├── pipeline_common.py         # Shared utilities
├── run_illustrated.py         # Illustrated pipeline entry point
├── run_visualizer.py          # Visualizer pipeline entry point
├── transcribe_words.py        # Word-level transcription
├── generate_prompts.py        # Semantic segmentation + prompt generation
├── batch_generate.py          # ComfyUI batch image generation
├── assemble_video.py          # Video assembly (all background modes)
└── output/
    ├── words.json             # Word-level timestamps from Whisper
    ├── transcript.txt         # Plain text transcript
    ├── prompts.jsonl          # Semantic chunks (+ image prompts if illustrated)
    ├── images/                # Generated images (illustrated only)
    │   ├── 0000.png
    │   └── ...
    └── lecture_video.mp4      # Final video
```

---

## How it works

| Step | Tool | What happens |
|------|------|-------------|
| Transcription | whisper-timestamped | Whisper ASR + DTW on cross-attention weights → per-word timestamps |
| Segmentation | Ollama (llama3) | LLM identifies idea boundaries in the transcript |
| Prompt generation | Ollama (llama3) | LLM generates metaphorical image descriptions per segment |
| Image generation | ComfyUI + SDXL | Latent diffusion with LoRA style control |
| Video assembly | Pillow + ffmpeg | Frame-by-frame rendering piped to H.264 encoder |

For animated backgrounds, steps 2–4 are simplified: only semantic segmentation runs (no image prompts or generation). The video assembler generates backgrounds procedurally each frame using FFT analysis (waveform/radial) or layered sine fields (lava).

---

## Troubleshooting

### PyTorch doesn't see my GPU

For RTX 50-series GPUs (Blackwell / sm_120), stable PyTorch releases don't include the required kernels yet. Use nightly builds:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### ComfyUI workflow nodes not detected

`batch_generate.py` auto-detects node IDs by scanning `class_type` fields in your `workflow_api.json`. If detection fails, make sure your workflow contains all required node types: `KSampler`, `CLIPTextEncode` (×2), `EmptyLatentImage`, `SaveImage`.

The script distinguishes positive/negative CLIP nodes by checking which ones the KSampler references. If that fails, it falls back to node title metadata (name your nodes "POS" and "NEG" in ComfyUI).

### Out of VRAM

- Use SDXL (not FLUX) for 8 GB cards
- The batch script processes images sequentially to avoid memory spikes
- Close ComfyUI's browser tab during batch generation (the preview uses VRAM)

### Video rendering is slow

Frame-by-frame rendering with Pillow is CPU-bound. For a 60-minute lecture at 30fps:
- `images` mode: ~10–20 minutes (just compositing)
- `waveform`/`radial` mode: ~20–40 minutes (FFT + drawing per frame)
- `lava` mode: ~15–25 minutes (numpy field computation per frame)

The `radial` mode renders at 2× resolution for antialiasing, which adds overhead.

---

## License

MIT
