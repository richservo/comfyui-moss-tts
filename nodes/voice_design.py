import torch
import comfy.model_management as mm

from ..utils.audio_utils import moss_tensor_to_comfyui_audio
from ..utils.backend import run_generation

VOICE_GENERATOR_MODEL_ID = "OpenMOSS-Team/MOSS-VoiceGenerator"


class MossTTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_pipe": ("MOSS_TTS_PIPE",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "instruction": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS-TTS"

    def generate(
        self,
        moss_pipe,
        text,
        instruction,
        seed,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
    ):
        model, processor, sample_rate, device, model_id = moss_pipe

        if model_id != VOICE_GENERATOR_MODEL_ID:
            print(
                f"[MOSS-TTS] Warning: MossTTSVoiceDesign expects model "
                f"'{VOICE_GENERATOR_MODEL_ID}' but got '{model_id}'. "
                "Results may be unexpected."
            )

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        user_msg = processor.build_user_message(
            text=text,
            instruction=instruction,
        )

        batch = processor([[user_msg]], mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = run_generation(
                model, input_ids, attention_mask, model_id, processor,
                temperature, top_p, top_k, repetition_penalty, max_new_tokens,
            )

        messages = processor.decode(outputs)
        if messages[0] is None:
            raise RuntimeError("Generation failed — model returned no audio")
        wav = messages[0].audio_codes_list[0]

        result = moss_tensor_to_comfyui_audio(wav.cpu(), sample_rate)

        mm.soft_empty_cache()
        return (result,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)
