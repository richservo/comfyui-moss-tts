import hashlib
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

import comfy.model_management as mm
import folder_paths

from ..utils.constants import MODEL_VARIANTS, DEFAULT_CODEC_PATH, SAMPLE_RATE
from ..utils import backend

MODEL_ID_TTSD = "OpenMOSS-Team/MOSS-TTSD-v1.0"
MODEL_ID_VOICE_GENERATOR = "OpenMOSS-Team/MOSS-VoiceGenerator"

# Store models in ComfyUI's models directory under moss-tts/
MOSS_MODELS_DIR = os.path.join(folder_paths.models_dir, "moss-tts")
os.makedirs(MOSS_MODELS_DIR, exist_ok=True)


def _resolve_local_dir(repo_id_or_path):
    """If repo_id_or_path is a HF repo ID, download it into ComfyUI's
    models/moss-tts/ directory and return the local path.
    If it's already a local path, return it as-is.
    Works around MOSS processor bug: Path(repo_id) mangles / to \\ on Windows."""
    if os.path.isdir(repo_id_or_path):
        return repo_id_or_path
    # e.g. "OpenMOSS-Team/MOSS-TTS" → "OpenMOSS-Team--MOSS-TTS"
    safe_name = repo_id_or_path.replace("/", "--")
    local_dir = os.path.join(MOSS_MODELS_DIR, safe_name)
    return snapshot_download(repo_id_or_path, local_dir=local_dir)


class MossTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (list(MODEL_VARIANTS.keys()),),
                "local_model_path": ("STRING", {"default": ""}),
                "codec_local_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MOSS_TTS_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "audio/MOSS-TTS"

    def load_model(self, model_variant, local_model_path, codec_local_path):
        device = mm.get_torch_device()
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        attn_implementation = backend.resolve_attn_implementation(device, dtype)

        mm.unload_all_models()

        model_id = MODEL_VARIANTS[model_variant]
        model_path = local_model_path.strip() if local_model_path.strip() else model_id

        # Pre-resolve to local directory to avoid MOSS processor's
        # Path(repo_id) bug on Windows (converts / to \)
        local_dir = _resolve_local_dir(model_path)

        processor_kwargs = {"trust_remote_code": True}
        if model_id == MODEL_ID_TTSD:
            codec_path = codec_local_path.strip() or DEFAULT_CODEC_PATH
            processor_kwargs["codec_path"] = _resolve_local_dir(codec_path)
        elif model_id == MODEL_ID_VOICE_GENERATOR:
            processor_kwargs["normalize_inputs"] = True

        processor = AutoProcessor.from_pretrained(local_dir, **processor_kwargs)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model = AutoModel.from_pretrained(
            local_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        return ((model, processor, SAMPLE_RATE, device, model_id),)

    @classmethod
    def IS_CHANGED(cls, model_variant, local_model_path, codec_local_path):
        h = hashlib.md5()
        h.update(model_variant.encode())
        h.update(local_model_path.encode())
        h.update(codec_local_path.encode())
        return h.hexdigest()
