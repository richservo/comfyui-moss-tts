# ComfyUI MOSS-TTS Custom Node Package — Implementation Plan

## Overview

Wrap the MOSS-TTS model family into a ComfyUI custom node package (`comfyui-moss-tts`) with 5 nodes covering: TTS, voice cloning, voice design, sound effects, and multi-speaker dialogue.

---

## File Structure

```
comfyui-moss-tts/
    __init__.py              # NODE_CLASS_MAPPINGS + NODE_DISPLAY_NAME_MAPPINGS
    requirements.txt         # Dependencies (no torch/torchaudio — ComfyUI provides)
    reference/
        plan.md              # This file
    nodes/
        __init__.py
        model_loader.py      # MossTTSModelLoader
        generate.py          # MossTTSGenerate (TTS + voice cloning)
        voice_design.py      # MossTTSVoiceDesign
        dialogue.py          # MossTTSDialogue (multi-speaker)
        sound_effect.py      # MossTTSSoundEffect
    utils/
        __init__.py
        backend.py           # cuDNN SDPA patch, attention resolution helper
        audio_utils.py       # ComfyUI AUDIO ↔ MOSS waveform conversion
        constants.py         # Model IDs, default params, duration constants
```

---

## Phase 0: GitHub Repo Setup

### Steps
1. `gh auth login` (if not already authenticated)
2. `gh repo create comfyui-moss-tts --private`
3. `cd /mnt/e/Python/MOSS-TTS/comfyui-moss-tts && git init && git remote add origin ...`
4. Create `.gitignore` (Python defaults + __pycache__, *.pyc, .venv, etc.)
5. Initial commit + push

### Verification
- `gh repo view comfyui-moss-tts` shows the repo
- `git log` shows initial commit

---

## Phase 1: Utility Modules

### 1a. `utils/constants.py`

Contains all model IDs, default hyperparameters, and the duration constant.

```python
# Model registry
MODEL_VARIANTS = {
    "MOSS-TTS (Delay 8B)":       "OpenMOSS-Team/MOSS-TTS",
    "MOSS-TTS (Local 1.7B)":     "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "MOSS-TTSD v1.0":            "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "MOSS-VoiceGenerator":       "OpenMOSS-Team/MOSS-VoiceGenerator",
    "MOSS-SoundEffect":          "OpenMOSS-Team/MOSS-SoundEffect",
}

DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

TOKENS_PER_SECOND = 12.5  # codec frame rate

# Per-model recommended defaults
DEFAULT_PARAMS = {
    "OpenMOSS-Team/MOSS-TTS": {
        "audio_temperature": 1.7, "audio_top_p": 0.8,
        "audio_top_k": 25, "audio_repetition_penalty": 1.0,
    },
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": {
        "audio_temperature": 1.0, "audio_top_p": 0.95,
        "audio_top_k": 50, "audio_repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-TTSD-v1.0": {
        "audio_temperature": 1.1, "audio_top_p": 0.9,
        "audio_top_k": 50, "audio_repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-VoiceGenerator": {
        "audio_temperature": 1.5, "audio_top_p": 0.6,
        "audio_top_k": 50, "audio_repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-SoundEffect": {
        "audio_temperature": 1.5, "audio_top_p": 0.6,
        "audio_top_k": 50, "audio_repetition_penalty": 1.2,
    },
}

MAX_NEW_TOKENS_DEFAULT = 4096
SAMPLE_RATE = 24000
```

### 1b. `utils/backend.py`

Applied once at import time. Contains:

```python
import torch
import importlib.util

# cuDNN SDPA patch — required by all MOSS models
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

def resolve_attn_implementation(device, dtype) -> str:
    """Pick best attention backend for this hardware."""
    if (
        str(device).startswith("cuda")
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"
```

### 1c. `utils/audio_utils.py`

Converts between ComfyUI `AUDIO` dict format and MOSS raw tensors.

```
ComfyUI AUDIO = {"waveform": Tensor([B, C, S]), "sample_rate": int}
MOSS expects: 1D float32 tensor at 24kHz (mono)
MOSS outputs: 1D float32 tensor at 24kHz
```

Key functions:
- `comfyui_audio_to_moss_tensor(audio_dict) -> (tensor_1d, sample_rate)` — squeeze to mono, return raw tensor
- `moss_tensor_to_comfyui_audio(tensor_1d, sample_rate=24000) -> dict` — reshape to [1, 1, S], return dict
- `resample_if_needed(waveform, orig_sr, target_sr) -> tensor` — uses torchaudio.functional.resample

### Verification (Phase 1)
- Import each module in Python to check for syntax errors
- `python -c "from utils.constants import MODEL_VARIANTS; print(MODEL_VARIANTS)"`
- `python -c "import utils.backend"` — should print nothing, just apply the patch
- `python -c "from utils.audio_utils import moss_tensor_to_comfyui_audio; import torch; r = moss_tensor_to_comfyui_audio(torch.randn(48000)); print(r['waveform'].shape, r['sample_rate'])"` — should print `torch.Size([1, 1, 48000]) 24000`

---

## Phase 2: Model Loader Node

### `nodes/model_loader.py` — `MossTTSModelLoader`

**Category:** `audio/MOSS-TTS`

**Inputs:**
| Name | Type | Default | Notes |
|------|------|---------|-------|
| model_variant | dropdown | first entry | Keys of `MODEL_VARIANTS` |
| local_model_path | STRING | "" | Override HF download with local path |
| codec_local_path | STRING | "" | Override codec download with local path |

**Output:** `MOSS_TTS_PIPE` tuple: `(model, processor, sample_rate, device, model_id)`

**Logic:**
1. Get device via `comfy.model_management.get_torch_device()`
2. Determine dtype: `torch.bfloat16` on CUDA, `torch.float32` otherwise
3. Resolve `attn_implementation` via `backend.resolve_attn_implementation()`
4. Call `mm.unload_all_models()` to free VRAM
5. Build `processor_kwargs`:
   - Always: `trust_remote_code=True`
   - If TTSD: add `codec_path` (from `codec_local_path` or `DEFAULT_CODEC_PATH`)
   - If VoiceGenerator: add `normalize_inputs=True`
6. `processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)`
7. `processor.audio_tokenizer = processor.audio_tokenizer.to(device)`
8. `model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation=..., torch_dtype=...).to(device)`
9. `model.eval()`
10. Return `(model, processor, SAMPLE_RATE, device, model_id)`

**`IS_CHANGED`:** Return `(model_variant, local_model_path, codec_local_path)` so ComfyUI re-runs only when inputs change.

### Verification (Phase 2)
- Symlink to ComfyUI custom_nodes
- Start ComfyUI, verify "MOSS-TTS Model Loader" appears in node search
- Add node to canvas, verify dropdown shows all 5 model variants
- Connect to nothing — just verify the UI renders correctly
- (Full load test deferred to Phase 3 since we need a generation node to verify output)

---

## Phase 3: Generate Node (TTS + Voice Cloning)

### `nodes/generate.py` — `MossTTSGenerate`

**Category:** `audio/MOSS-TTS`

**Inputs:**
| Name | Type | Default | Range | Required |
|------|------|---------|-------|----------|
| moss_pipe | MOSS_TTS_PIPE | — | — | yes |
| text | STRING (multiline) | "" | — | yes |
| seed | INT | 0 | [0, 2^31-1] | yes |
| temperature | FLOAT | 1.7 | [0.01, 5.0] | yes |
| top_p | FLOAT | 0.8 | [0.0, 1.0] | yes |
| top_k | INT | 25 | [1, 500] | yes |
| repetition_penalty | FLOAT | 1.0 | [0.0, 5.0] | yes |
| max_new_tokens | INT | 4096 | [64, 16384] | yes |
| enable_duration_control | BOOLEAN | False | — | yes |
| duration_tokens | INT | 325 | [1, 16384] | yes |
| reference_audio | AUDIO | None | — | optional |

**Output:** `AUDIO`

**Logic:**
1. Unpack `moss_pipe`
2. `torch.manual_seed(seed)`
3. Build reference list:
   - If `reference_audio` provided: convert via `comfyui_audio_to_moss_tensor()`, resample to 24kHz, encode via `processor.encode_audios_from_wav([tensor], sample_rate)`
   - Else: `reference = None`
4. Build `tokens` param: if `enable_duration_control`, use `duration_tokens`; else `None`
5. Build conversation: `processor.build_user_message(text=text, reference=reference, tokens=tokens)`
6. Tokenize: `batch = processor([[user_msg]], mode="generation")`
7. Generate:
   ```python
   outputs = model.generate(
       input_ids=batch["input_ids"].to(device),
       attention_mask=batch["attention_mask"].to(device),
       max_new_tokens=max_new_tokens,
       audio_temperature=temperature,
       audio_top_p=top_p,
       audio_top_k=top_k,
       audio_repetition_penalty=repetition_penalty,
   )
   ```
8. Decode: `messages = processor.decode(outputs)`
9. Extract waveform: `wav = messages[0].audio_codes_list[0]`
10. Convert to ComfyUI AUDIO: `moss_tensor_to_comfyui_audio(wav, SAMPLE_RATE)`
11. `mm.soft_empty_cache()` after generation

**`IS_CHANGED`:** Return `seed`

### Verification (Phase 3)
- Restart ComfyUI, verify "MOSS-TTS Generate" appears
- **Test A — Direct TTS:** ModelLoader (Local 1.7B) → Generate (text="Hello world, this is a test.") → PreviewAudio
  - Verify audio plays, is intelligible speech
- **Test B — Voice Cloning:** LoadAudio → Generate (with reference_audio connected) → PreviewAudio
  - Verify output mimics reference voice
- **Test C — Duration Control:** Enable duration_control, set tokens=125 (10 sec), verify output length is ~10s

---

## Phase 4: Voice Design Node

### `nodes/voice_design.py` — `MossTTSVoiceDesign`

**Category:** `audio/MOSS-TTS`

**Inputs:**
| Name | Type | Default | Range | Required |
|------|------|---------|-------|----------|
| moss_pipe | MOSS_TTS_PIPE | — | — | yes |
| text | STRING (multiline) | "" | — | yes |
| instruction | STRING (multiline) | "" | — | yes |
| seed | INT | 0 | [0, 2^31-1] | yes |
| temperature | FLOAT | 1.5 | [0.01, 5.0] | yes |
| top_p | FLOAT | 0.6 | [0.0, 1.0] | yes |
| top_k | INT | 50 | [1, 500] | yes |
| repetition_penalty | FLOAT | 1.1 | [0.0, 5.0] | yes |
| max_new_tokens | INT | 4096 | [64, 16384] | yes |

**Output:** `AUDIO`

**Logic:**
1. Unpack `moss_pipe`, validate `model_id == "OpenMOSS-Team/MOSS-VoiceGenerator"` (warn but don't block if mismatched)
2. `torch.manual_seed(seed)`
3. `user_msg = processor.build_user_message(text=text, instruction=instruction)`
4. Same tokenize → generate → decode → convert flow as Generate node
5. `mm.soft_empty_cache()`

### Verification (Phase 4)
- Restart ComfyUI, verify "MOSS-TTS Voice Design" appears
- **Test:** ModelLoader (VoiceGenerator) → VoiceDesign (text="The quick brown fox jumps over the lazy dog.", instruction="A warm, deep male voice with a slight British accent, speaking at a moderate pace.") → PreviewAudio
- Verify output matches the voice description

---

## Phase 5: Sound Effect Node

### `nodes/sound_effect.py` — `MossTTSSoundEffect`

**Category:** `audio/MOSS-TTS`

**Inputs:**
| Name | Type | Default | Range | Required |
|------|------|---------|-------|----------|
| moss_pipe | MOSS_TTS_PIPE | — | — | yes |
| ambient_sound | STRING (multiline) | "" | — | yes |
| duration_seconds | FLOAT | 5.0 | [0.5, 60.0] | yes |
| seed | INT | 0 | [0, 2^31-1] | yes |
| temperature | FLOAT | 1.5 | [0.01, 5.0] | yes |
| top_p | FLOAT | 0.6 | [0.0, 1.0] | yes |
| top_k | INT | 50 | [1, 500] | yes |
| repetition_penalty | FLOAT | 1.2 | [0.0, 5.0] | yes |
| max_new_tokens | INT | 4096 | [64, 16384] | yes |

**Output:** `AUDIO`

**Logic:**
1. Unpack `moss_pipe`, validate `model_id == "OpenMOSS-Team/MOSS-SoundEffect"`
2. `torch.manual_seed(seed)`
3. Convert duration: `tokens = max(1, int(duration_seconds * TOKENS_PER_SECOND))`
4. `user_msg = processor.build_user_message(ambient_sound=ambient_sound, tokens=tokens)`
5. Same tokenize → generate → decode → convert flow
6. `mm.soft_empty_cache()`

### Verification (Phase 5)
- Restart ComfyUI, verify "MOSS-TTS Sound Effect" appears
- **Test:** ModelLoader (SoundEffect) → SoundEffect (ambient_sound="Heavy rain falling on a tin roof with distant thunder", duration=5.0) → PreviewAudio
- Verify output sounds like rain/thunder, is approximately 5 seconds

---

## Phase 6: Dialogue Node

### `nodes/dialogue.py` — `MossTTSDialogue`

**Category:** `audio/MOSS-TTS`

This is the most complex node. It handles multi-speaker dialogue with optional per-speaker voice cloning.

**Inputs:**
| Name | Type | Default | Range | Required |
|------|------|---------|-------|----------|
| moss_pipe | MOSS_TTS_PIPE | — | — | yes |
| dialogue_text | STRING (multiline) | "" | — | yes |
| speaker_count | INT | 2 | [2, 2] | yes |
| normalize_text | BOOLEAN | True | — | yes |
| seed | INT | 0 | [0, 2^31-1] | yes |
| temperature | FLOAT | 1.1 | [0.01, 5.0] | yes |
| top_p | FLOAT | 0.9 | [0.0, 1.0] | yes |
| top_k | INT | 50 | [1, 500] | yes |
| repetition_penalty | FLOAT | 1.1 | [0.0, 5.0] | yes |
| max_new_tokens | INT | 4096 | [64, 16384] | yes |
| s1_reference_audio | AUDIO | None | — | optional |
| s1_prompt_text | STRING | "" | — | optional |
| s2_reference_audio | AUDIO | None | — | optional |
| s2_prompt_text | STRING | "" | — | optional |

**Output:** `AUDIO`

**Logic:**
1. Unpack `moss_pipe`, validate `model_id == "OpenMOSS-Team/MOSS-TTSD-v1.0"`
2. `torch.manual_seed(seed)`
3. Text normalization (if enabled): ensure dialogue text has proper `[S1]`/`[S2]` tags
4. Build reference list for each speaker:
   - For each speaker with reference audio: convert ComfyUI AUDIO → mono tensor → resample to 24kHz → `processor.encode_audios_from_wav()`
   - Speakers without reference: `None` in the reference list
5. Build `user_msg = processor.build_user_message(text=dialogue_text, reference=references)`
6. If any speaker has reference audio AND prompt_text, use continuation mode:
   - Concatenate prompt audio codes
   - `assistant_msg = processor.build_assistant_message(audio_codes_list=[prompt_codes])`
   - `conversations = [[user_msg, assistant_msg]]`
   - `mode = "continuation"`
7. Else: `conversations = [[user_msg]]`, `mode = "generation"`
8. Tokenize → generate → decode → convert
9. `mm.soft_empty_cache()`

### Verification (Phase 6)
- Restart ComfyUI, verify "MOSS-TTS Dialogue" appears
- **Test A — No references:** ModelLoader (TTSD) → Dialogue (text="[S1] Hello, how are you?\n[S2] I'm doing great, thanks for asking!\n[S1] That's wonderful to hear.") → PreviewAudio
  - Verify two distinct speaker voices
- **Test B — With references:** LoadAudio (speaker 1) + LoadAudio (speaker 2) → Dialogue (with both references) → PreviewAudio
  - Verify speakers match references

---

## Phase 7: Package Integration

### 7a. Top-level `__init__.py`

```python
from .nodes.model_loader import MossTTSModelLoader
from .nodes.generate import MossTTSGenerate
from .nodes.voice_design import MossTTSVoiceDesign
from .nodes.sound_effect import MossTTSSoundEffect
from .nodes.dialogue import MossTTSDialogue

NODE_CLASS_MAPPINGS = {
    "MossTTSModelLoader": MossTTSModelLoader,
    "MossTTSGenerate": MossTTSGenerate,
    "MossTTSVoiceDesign": MossTTSVoiceDesign,
    "MossTTSSoundEffect": MossTTSSoundEffect,
    "MossTTSDialogue": MossTTSDialogue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSModelLoader": "MOSS-TTS Model Loader",
    "MossTTSGenerate": "MOSS-TTS Generate",
    "MossTTSVoiceDesign": "MOSS-TTS Voice Design",
    "MossTTSSoundEffect": "MOSS-TTS Sound Effect",
    "MossTTSDialogue": "MOSS-TTS Dialogue",
}
```

### 7b. `requirements.txt`

```
transformers>=4.40.0
safetensors
einops
scipy
librosa
tiktoken
```

No `torch` / `torchaudio` — ComfyUI provides these.

### Verification (Phase 7)
- Symlink: `ln -s /mnt/e/Python/MOSS-TTS/comfyui-moss-tts /mnt/c/Users/Richard/Documents/ComfyUI/custom_nodes/comfyui-moss-tts`
- Start ComfyUI, check console for import errors
- All 5 nodes appear under "audio/MOSS-TTS" category
- Create a minimal workflow: ModelLoader → Generate → PreviewAudio
- Run it end-to-end

---

## Phase 8: Git Commit & Push

### Steps
1. Stage all files
2. Commit with descriptive message
3. Push to GitHub

---

## API Reference (from repo analysis)

### Processor API

```python
# Build messages
processor.build_user_message(
    text=None, reference=None, instruction=None,
    tokens=None, quality=None, sound_event=None,
    ambient_sound=None, language=None,
)
processor.build_assistant_message(audio_codes_list=[...], content="<|audio|>")

# Tokenize
batch = processor(conversations, mode="generation"|"continuation", apply_chat_template=True)
# Returns: {"input_ids": Tensor[B, T, 1+n_vq], "attention_mask": Tensor[B, T]}

# Decode outputs
messages = processor.decode(outputs)
# messages[i].audio_codes_list[0] → 1D float32 waveform tensor

# Audio encode/decode
codes = processor.encode_audios_from_wav(wav_list, sampling_rate, n_vq=None)
codes = processor.encode_audios_from_path(path_list, n_vq=None)
wavs = processor.decode_audio_codes(audio_tokens_list)
```

### Model Generate API (Delay 8B)

```python
outputs = model.generate(
    input_ids,               # (B, T, 1+n_vq)
    attention_mask=None,     # (B, T)
    max_new_tokens=1000,
    text_temperature=1.5,
    text_top_p=1.0,
    text_top_k=50,
    audio_temperature=1.7,
    audio_top_p=0.8,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
)
# Returns: List[Tuple[int, Tensor]]
```

### Model Generate API (Local 1.7B)

Uses HuggingFace `GenerationMixin` with custom `_sample()`. Parameters passed via `generation_config`:
```python
generation_config.layers = [{"temperature": ..., "top_k": ..., "top_p": ..., "repetition_penalty": ...}]
generation_config.do_samples = [True/False per channel]
```

### ComfyUI AUDIO Format
```python
{"waveform": Tensor([batch, channels, samples]), "sample_rate": int}
# Standard: batch=1, channels=1, sample_rate=24000
```

### HuggingFace Model IDs
| Display Name | Model ID |
|---|---|
| MOSS-TTS (Delay 8B) | `OpenMOSS-Team/MOSS-TTS` |
| MOSS-TTS (Local 1.7B) | `OpenMOSS-Team/MOSS-TTS-Local-Transformer` |
| MOSS-TTSD v1.0 | `OpenMOSS-Team/MOSS-TTSD-v1.0` |
| MOSS-VoiceGenerator | `OpenMOSS-Team/MOSS-VoiceGenerator` |
| MOSS-SoundEffect | `OpenMOSS-Team/MOSS-SoundEffect` |
| Audio Tokenizer (codec) | `OpenMOSS-Team/MOSS-Audio-Tokenizer` |

### Recommended Hyperparameters
| Model | temp | top_p | top_k | rep_penalty |
|---|---|---|---|---|
| Delay 8B | 1.7 | 0.8 | 25 | 1.0 |
| Local 1.7B | 1.0 | 0.95 | 50 | 1.1 |
| TTSD | 1.1 | 0.9 | 50 | 1.1 |
| VoiceGenerator | 1.5 | 0.6 | 50 | 1.1 |
| SoundEffect | 1.5 | 0.6 | 50 | 1.2 |

### Key Constants
- Sample rate: **24000 Hz**
- Codec frame rate: **12.5 tokens/second**
- Duration formula: `tokens = int(seconds * 12.5)`
