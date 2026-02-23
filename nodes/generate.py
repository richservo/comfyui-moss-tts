import torch
import comfy.model_management as mm

from ..utils.audio_utils import (
    apply_handles,
    comfyui_audio_to_moss_tensor,
    moss_tensor_to_comfyui_audio,
    resample_if_needed,
)
from ..utils.backend import run_generation


class MossTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_pipe": ("MOSS_TTS_PIPE",),
                "language": (["auto", "zh", "en", "ja", "ko"],),
                "text": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "enable_duration_control": ("BOOLEAN", {"default": False}),
                "duration_tokens": ("INT", {"default": 325, "min": 1, "max": 4096, "step": 1}),
                "head_handle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tail_handle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS-TTS"

    def generate(
        self,
        moss_pipe,
        language,
        text,
        seed,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
        enable_duration_control,
        duration_tokens,
        head_handle,
        tail_handle,
        reference_audio=None,
    ):
        model, processor, sample_rate, device, model_id = moss_pipe

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        reference = None
        if reference_audio is not None:
            wav, orig_sr = comfyui_audio_to_moss_tensor(reference_audio)
            wav = resample_if_needed(wav, orig_sr, sample_rate)
            # encode_audios_from_wav expects 2D tensors [channels, samples]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            codes_list = processor.encode_audios_from_wav(
                wav_list=[wav],
                sampling_rate=sample_rate,
            )
            reference = codes_list

        tokens = duration_tokens if enable_duration_control else None

        lang = None if language == "auto" else language

        user_msg = processor.build_user_message(
            text=text,
            reference=reference,
            tokens=tokens,
            language=lang,
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

        wav = apply_handles(wav.cpu(), sample_rate, head_handle, tail_handle)
        result = moss_tensor_to_comfyui_audio(wav, sample_rate)

        mm.soft_empty_cache()
        return (result,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)
