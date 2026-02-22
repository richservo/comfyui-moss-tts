# ComfyUI MOSS-TTS

Custom nodes for the [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) model family from OpenMOSS. Brings text-to-speech, zero-shot voice cloning, voice design, sound effect generation, and multi-speaker dialogue synthesis into ComfyUI workflows.

<!-- TODO: Add screenshot -->

---

## Features

- **Text-to-speech**: Generate natural speech from text using the base MOSS-TTS models (8B Delay or 1.7B Local)
- **Zero-shot voice cloning**: Clone any voice from a 3-10 second reference clip, no transcript required
- **Voice design**: Describe a voice in plain language and generate speech in that style (requires MOSS-VoiceGenerator)
- **Sound effect generation**: Generate ambient sounds and audio effects from text descriptions with controllable duration (requires MOSS-SoundEffect)
- **Multi-speaker dialogue**: Synthesize conversations between two speakers with optional per-speaker voice cloning (requires MOSS-TTSD)
- **Duration control**: Target a specific output length using token-based duration hints
- **All models output 24 kHz audio** compatible with ComfyUI's native AUDIO type

---

## Installation

**1. Clone into your ComfyUI custom nodes directory:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/richservo/comfyui-moss-tts
```

**2. Install dependencies:**

```bash
pip install -r ComfyUI/custom_nodes/comfyui-moss-tts/requirements.txt
```

> **Note:** `transformers>=5.0.0` is required. Earlier versions may fail to load the MOSS model architecture. `torch` and `torchaudio` are not listed in requirements because ComfyUI provides them.

**3. Restart ComfyUI.**

All five nodes will appear under the **audio/MOSS-TTS** category in the node search.

Models download automatically from HuggingFace on first use and are cached to `ComfyUI/models/moss-tts/`.

---

## Models

| Display Name | HuggingFace ID | Architecture | Approx. VRAM | Speed Notes |
|---|---|---|---|---|
| MOSS-TTS (Delay 8B) | `OpenMOSS-Team/MOSS-TTS` | Delay 8B | ~18 GB | Slow on single GPU |
| MOSS-TTS (Local 1.7B) | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` | Local 1.7B | ~5 GB | Fast; recommended for daily use |
| MOSS-TTSD v1.0 | `OpenMOSS-Team/MOSS-TTSD-v1.0` | Delay 8B | ~18 GB | Slow on single GPU |
| MOSS-VoiceGenerator | `OpenMOSS-Team/MOSS-VoiceGenerator` | Delay 8B | ~18 GB | Slow on single GPU |
| MOSS-SoundEffect | `OpenMOSS-Team/MOSS-SoundEffect` | Delay 8B | ~18 GB | Slow on single GPU |

**MOSS-TTS (Local 1.7B)** is the only model that is fast enough for practical iterative use on a single consumer GPU. All other models use the Delay 8B architecture and are significantly slower.

The **Audio Tokenizer** (`OpenMOSS-Team/MOSS-Audio-Tokenizer`) is a shared codec required by all models. It downloads automatically alongside whichever model you load first.

---

## Nodes

### MOSS-TTS Model Loader

Loads a MOSS-TTS model and processor into a pipeline object (`MOSS_TTS_PIPE`) that is passed to any generation node.

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `model_variant` | Dropdown | MOSS-TTS (Delay 8B) | Which model to load |
| `local_model_path` | String | _(empty)_ | Absolute path to a local model directory. Leave empty to auto-download from HuggingFace |
| `codec_local_path` | String | _(empty)_ | Absolute path to a local Audio Tokenizer directory (relevant for MOSS-TTSD). Leave empty to auto-download |

**Output:** `MOSS_TTS_PIPE`

**Notes:**
- Calls `mm.unload_all_models()` before loading to free VRAM.
- Models are cached to `ComfyUI/models/moss-tts/` on first download.
- The node uses an MD5 hash of its inputs to detect changes and avoid unnecessary reloads.

---

### MOSS-TTS Generate

Generates speech from text. Optionally clones a voice from a reference audio clip.

<!-- TODO: Add screenshot -->

**Inputs:**

| Name | Type | Default | Range | Description |
|---|---|---|---|---|
| `moss_pipe` | MOSS_TTS_PIPE | — | — | Pipeline from Model Loader |
| `text` | String (multiline) | _(empty)_ | — | Text to synthesize |
| `seed` | Int | 0 | 0 – 2^64 | Random seed for reproducibility |
| `temperature` | Float | 1.7 | 0.0 – 5.0 | Sampling temperature (default tuned for Delay 8B; use 1.0 for Local 1.7B) |
| `top_p` | Float | 0.8 | 0.0 – 1.0 | Nucleus sampling probability |
| `top_k` | Int | 25 | 1 – 200 | Top-k sampling (default tuned for Delay 8B; use 50 for Local 1.7B) |
| `repetition_penalty` | Float | 1.0 | 0.5 – 2.0 | Penalizes repeated tokens |
| `max_new_tokens` | Int | 4096 | 1 – 8192 | Maximum tokens to generate |
| `enable_duration_control` | Boolean | False | — | Enable target duration hint |
| `duration_tokens` | Int | 325 | 1 – 4096 | Target duration in tokens (1 second = 12.5 tokens, so 325 = ~26 seconds) |
| `reference_audio` | AUDIO | _(optional)_ | — | Reference audio for voice cloning |

**Output:** `AUDIO`

**Notes:**
- Without `reference_audio`: performs standard TTS with the model's default voice.
- With `reference_audio`: performs zero-shot voice cloning. No reference transcript is needed (unlike Qwen-TTS).
- The node re-runs whenever `seed` changes.

---

### MOSS-TTS Voice Design

Generates speech in a voice described by a natural language instruction. Requires the **MOSS-VoiceGenerator** model.

<!-- TODO: Add screenshot -->

**Inputs:**

| Name | Type | Default | Range | Description |
|---|---|---|---|---|
| `moss_pipe` | MOSS_TTS_PIPE | — | — | Pipeline from Model Loader (use MOSS-VoiceGenerator) |
| `text` | String (multiline) | _(empty)_ | — | Text to speak |
| `instruction` | String (multiline) | _(empty)_ | — | Natural language voice description (e.g. "A warm, deep male voice with a slight British accent") |
| `seed` | Int | 0 | 0 – 2^64 | Random seed |
| `temperature` | Float | 1.5 | 0.0 – 5.0 | Sampling temperature |
| `top_p` | Float | 0.6 | 0.0 – 1.0 | Nucleus sampling probability |
| `top_k` | Int | 50 | 1 – 200 | Top-k sampling |
| `repetition_penalty` | Float | 1.1 | 0.5 – 2.0 | Penalizes repeated tokens |
| `max_new_tokens` | Int | 4096 | 1 – 8192 | Maximum tokens to generate |

**Output:** `AUDIO`

**Notes:**
- The node will print a warning if the loaded model is not `MOSS-VoiceGenerator`, but will still attempt generation.

---

### MOSS-TTS Sound Effect

Generates ambient sounds and audio effects from a text description. Requires the **MOSS-SoundEffect** model.

<!-- TODO: Add screenshot -->

**Inputs:**

| Name | Type | Default | Range | Description |
|---|---|---|---|---|
| `moss_pipe` | MOSS_TTS_PIPE | — | — | Pipeline from Model Loader (use MOSS-SoundEffect) |
| `ambient_sound` | String (multiline) | _(empty)_ | — | Description of the sound to generate (e.g. "Heavy rain on a tin roof with distant thunder") |
| `duration_seconds` | Float | 5.0 | 0.5 – 60.0 | Target duration of the output audio in seconds |
| `seed` | Int | 0 | 0 – 2^64 | Random seed |
| `temperature` | Float | 1.5 | 0.0 – 5.0 | Sampling temperature |
| `top_p` | Float | 0.6 | 0.0 – 1.0 | Nucleus sampling probability |
| `top_k` | Int | 50 | 1 – 200 | Top-k sampling |
| `repetition_penalty` | Float | 1.2 | 0.5 – 2.0 | Penalizes repeated tokens |
| `max_new_tokens` | Int | 4096 | 1 – 8192 | Maximum tokens to generate |

**Output:** `AUDIO`

**Notes:**
- Duration is converted to tokens internally: `tokens = int(duration_seconds * 12.5)`.
- The node will print a warning if the loaded model is not `MOSS-SoundEffect`.

---

### MOSS-TTS Dialogue

Synthesizes multi-speaker dialogue with two distinct voices. Supports optional per-speaker voice cloning from reference audio. Requires the **MOSS-TTSD v1.0** model.

<!-- TODO: Add screenshot -->

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `moss_pipe` | MOSS_TTS_PIPE | — | Pipeline from Model Loader (use MOSS-TTSD v1.0) |
| `dialogue_text` | String (multiline) | _(empty)_ | Dialogue with speaker tags (see format below) |
| `speaker_count` | Int (fixed) | 2 | Fixed at 2; reserved for future expansion |
| `normalize_text` | Boolean | True | Apply text normalization (punctuation cleanup, tag normalization) |
| `seed` | Int | 0 | Random seed |
| `temperature` | Float | 1.1 | Sampling temperature |
| `top_p` | Float | 0.9 | Nucleus sampling probability |
| `top_k` | Int | 50 | Top-k sampling |
| `repetition_penalty` | Float | 1.1 | Penalizes repeated tokens |
| `max_new_tokens` | Int | 4096 | Maximum tokens to generate |
| `s1_reference_audio` | AUDIO | _(optional)_ | Reference clip for Speaker 1 voice cloning |
| `s1_prompt_text` | String | _(optional)_ | Transcript of the Speaker 1 reference clip |
| `s2_reference_audio` | AUDIO | _(optional)_ | Reference clip for Speaker 2 voice cloning |
| `s2_prompt_text` | String | _(optional)_ | Transcript of the Speaker 2 reference clip |

**Output:** `AUDIO`

**Dialogue text format:**

```
[S1] Hello, how are you today?
[S2] I'm doing great, thanks for asking!
[S1] That's wonderful to hear.
```

Accepted tag formats: `[S1]`/`[S2]`, `[s1]`/`[s2]`, or `[1]`/`[2]`. When `normalize_text` is enabled, all formats are normalized to `[S1]`/`[S2]` automatically.

**Notes:**
- Without reference audio: the model assigns random but distinct voices to each speaker.
- With reference audio for one or both speakers: the node uses continuation mode, conditioning generation on the reference clips.
- `s1_prompt_text` / `s2_prompt_text` are used as the continuation prefix transcript. If left empty, the reference audio is still used for voice conditioning.
- The node will print a warning if the loaded model is not `MOSS-TTSD-v1.0`.

---

## Example Workflows

### Basic TTS

```
MOSS-TTS Model Loader  →  MOSS-TTS Generate  →  PreviewAudio
```

Load the **MOSS-TTS (Local 1.7B)** model for fast iteration. Enter text in the Generate node and queue the prompt.

### Voice Cloning

```
LoadAudio (reference clip)  ──┐
                               ├─→  MOSS-TTS Generate  →  PreviewAudio
MOSS-TTS Model Loader  ───────┘
```

Connect a short audio clip (3-10 seconds) to the `reference_audio` input of the Generate node. The model will clone the voice without requiring a transcript.

### Voice Design

```
MOSS-TTS Model Loader (VoiceGenerator)  →  MOSS-TTS Voice Design  →  PreviewAudio
```

Load the **MOSS-VoiceGenerator** model. Describe the desired voice in the `instruction` field and provide the text to speak.

### Sound Effects

```
MOSS-TTS Model Loader (SoundEffect)  →  MOSS-TTS Sound Effect  →  PreviewAudio
```

Load the **MOSS-SoundEffect** model. Describe the desired sound in `ambient_sound` and set `duration_seconds`.

### Multi-Speaker Dialogue

```
MOSS-TTS Model Loader (TTSD)  →  MOSS-TTS Dialogue  →  PreviewAudio
```

Load the **MOSS-TTSD v1.0** model. Write dialogue in the `dialogue_text` field using `[S1]`/`[S2]` tags. Optionally connect reference audio clips for each speaker to clone their voices.

---

## Tips

**Reference audio quality:**
- The optimal reference clip length is **3-10 seconds**.
- Use clean recordings with a single speaker and no background noise.
- Clips longer than ~15 seconds may introduce noise artifacts or degrade quality.

**Model selection:**
- Use **MOSS-TTS (Local 1.7B)** for all day-to-day work — it is the only model fast enough for iterative use on a single GPU.
- Use the **Delay 8B** models (base TTS, TTSD, VoiceGenerator, SoundEffect) when maximum quality is the priority and generation time is not a concern.

**Sampling parameters:**
- Each node ships with defaults tuned for its intended model. If you switch models, refer to the recommended hyperparameters table below.
- Lowering temperature produces more consistent but less expressive output. Raising it increases variety but may introduce instability.

**Recommended parameters per model:**

| Model | temperature | top_p | top_k | repetition_penalty |
|---|---|---|---|---|
| MOSS-TTS (Delay 8B) | 1.7 | 0.8 | 25 | 1.0 |
| MOSS-TTS (Local 1.7B) | 1.0 | 0.95 | 50 | 1.1 |
| MOSS-TTSD v1.0 | 1.1 | 0.9 | 50 | 1.1 |
| MOSS-VoiceGenerator | 1.5 | 0.6 | 50 | 1.1 |
| MOSS-SoundEffect | 1.5 | 0.6 | 50 | 1.2 |

**Duration control:**
- All models use a codec frame rate of **12.5 tokens per second**.
- Formula: `tokens = int(seconds * 12.5)` — so 325 tokens ≈ 26 seconds, 125 tokens ≈ 10 seconds.
- Duration control is a hint, not a hard limit. Actual output length may vary.

**Audio output:**
- All models output **24 kHz mono audio**.
- Output is a standard ComfyUI `AUDIO` type and is compatible with all built-in ComfyUI audio nodes (PreviewAudio, SaveAudio, etc.).

**Voice cloning vs. Qwen-TTS:**
- MOSS-TTS does **not** require a reference transcript for voice cloning. Connecting a reference clip to `reference_audio` is sufficient.

**Dialogue tags:**
- The Dialogue node accepts `[S1]`/`[S2]`, `[s1]`/`[s2]`, or `[1]`/`[2]` — all are normalized automatically when `normalize_text` is enabled.
- Keep each speaker turn on its own line for best results.

---

## Credits

MOSS-TTS is developed by the [OpenMOSS](https://github.com/OpenMOSS) team.

- MOSS-TTS GitHub repository: [https://github.com/OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- HuggingFace organization: [https://huggingface.co/OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team)

This ComfyUI node package is an independent wrapper and is not officially affiliated with OpenMOSS.
