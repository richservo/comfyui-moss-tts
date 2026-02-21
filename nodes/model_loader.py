import hashlib
import torch
from transformers import AutoModel, AutoProcessor

import comfy.model_management as mm

from ..utils.constants import MODEL_VARIANTS, DEFAULT_CODEC_PATH, SAMPLE_RATE
from ..utils import backend

MODEL_ID_TTSD = "OpenMOSS-Team/MOSS-TTSD-v1.0"
MODEL_ID_VOICE_GENERATOR = "OpenMOSS-Team/MOSS-VoiceGenerator"


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
        model_path = local_model_path if local_model_path.strip() else model_id

        processor_kwargs = {"trust_remote_code": True}
        if model_id == MODEL_ID_TTSD:
            processor_kwargs["codec_path"] = codec_local_path.strip() or DEFAULT_CODEC_PATH
        elif model_id == MODEL_ID_VOICE_GENERATOR:
            processor_kwargs["normalize_inputs"] = True

        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model = AutoModel.from_pretrained(
            model_path,
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
