import re

import torch
import comfy.model_management as mm

from ..utils.audio_utils import (
    comfyui_audio_to_moss_tensor,
    moss_tensor_to_comfyui_audio,
    resample_if_needed,
)

TTSD_MODEL_ID = "OpenMOSS-Team/MOSS-TTSD-v1.0"


def _normalize_text(text: str) -> str:
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
    remove_chars = "【】《》（）『』「」\u2018\u2019\u201c\u201d\"-_\u201c\u201d\uff5e~\u2018\u2019"

    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = matched.groups() if matched else ("", seg)

        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"\u54c8{2,}", "[\u7b11]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)

        content = content.replace("\u2014\u2014", "\uff0c")
        content = content.replace("\u2026\u2026", "\uff0c")
        content = content.replace("...", "\uff0c")
        content = content.replace("\u2e3a", "\uff0c")
        content = content.replace("\u2015", "\uff0c")
        content = content.replace("\u2014", "\uff0c")
        content = content.replace("\u2026", "\uff0c")

        internal_punct_map = str.maketrans(
            {"\uff1b": "\uff0c", ";": ",", "\uff1a": "\uff0c", ":": ",", "\u3001": "\uff0c"}
        )
        content = content.translate(internal_punct_map)
        content = content.strip()
        content = re.sub(r"([\uff0c\u3002\uff1f\uff01,.?!])[\uff0c\u3002\uff1f\uff01,.?!]+", r"\1", content)

        if len(content) > 1:
            last_ch = "\u3002" if content[-1] == "\uff0c" else ("." if content[-1] == "," else content[-1])
            body = content[:-1].replace("\u3002", "\uff0c")
            content = body + last_ch

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        return ""

    merged_lines = []
    current_tag = processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]
    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]
    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

    return "".join(merged_lines).replace("\u2018", "'").replace("\u2019", "'")


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    merged_parts = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if not matched:
            merged_parts.append(seg)
            continue
        tag, content = matched.groups()
        if tag == current_tag:
            merged_parts.append(content)
        else:
            current_tag = tag
            merged_parts.append(f"{tag}{content}")
    return "".join(merged_parts)


def _normalize_prompt_text(prompt_text: str, speaker_id: int) -> str:
    text = prompt_text.strip()
    expected_tag = f"[S{speaker_id}]"
    if not text.lstrip().startswith(expected_tag):
        text = f"{expected_tag} {text}"
    return text


class MossTTSDialogue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_pipe": ("MOSS_TTS_PIPE",),
                "dialogue_text": ("STRING", {"default": "", "multiline": True}),
                "speaker_count": ("INT", {"default": 2, "min": 2, "max": 2, "step": 1}),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "s1_reference_audio": ("AUDIO",),
                "s1_prompt_text": ("STRING", {"default": "", "multiline": False}),
                "s2_reference_audio": ("AUDIO",),
                "s2_prompt_text": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS-TTS"

    def generate(
        self,
        moss_pipe,
        dialogue_text,
        speaker_count,
        normalize_text,
        seed,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
        s1_reference_audio=None,
        s1_prompt_text="",
        s2_reference_audio=None,
        s2_prompt_text="",
    ):
        model, processor, sample_rate, device, model_id = moss_pipe

        if model_id != TTSD_MODEL_ID:
            print(
                f"[MOSS-TTS] Warning: MossTTSDialogue expects model '{TTSD_MODEL_ID}', "
                f"but got '{model_id}'. Results may be unexpected."
            )

        torch.manual_seed(seed)

        # Normalise and validate the dialogue text
        text = dialogue_text.strip()
        if normalize_text:
            text = _normalize_text(text)

        # Gather per-speaker inputs (indexed 1-based to match [S1]/[S2] tags)
        ref_audios = [s1_reference_audio, s2_reference_audio]
        prompt_texts = [s1_prompt_text or "", s2_prompt_text or ""]

        # Determine which speakers have reference audio
        cloned_speakers: list[int] = []      # 1-based speaker IDs
        clone_wavs: list[torch.Tensor] = []  # resampled mono tensors, one per cloned speaker
        prompt_text_map: dict[int, str] = {} # normalised prompt text per speaker

        for idx in range(speaker_count):
            speaker_id = idx + 1
            ref_audio = ref_audios[idx]
            prompt_text = prompt_texts[idx].strip()

            if ref_audio is None:
                continue

            wav, orig_sr = comfyui_audio_to_moss_tensor(ref_audio)
            wav = resample_if_needed(wav, orig_sr, sample_rate)
            cloned_speakers.append(speaker_id)
            clone_wavs.append(wav)
            prompt_text_map[speaker_id] = _normalize_prompt_text(prompt_text, speaker_id)

        if not cloned_speakers:
            # Generation mode — no reference audio for any speaker
            user_msg = processor.build_user_message(text=text)
            conversations = [[user_msg]]
            mode = "generation"
        else:
            # Build conversation_text: prepend each speaker's prompt text, then the dialogue
            prompt_prefix = "".join(prompt_text_map[sid] for sid in cloned_speakers)
            conversation_text = _merge_consecutive_speaker_tags(prompt_prefix + text)
            if normalize_text:
                conversation_text = _normalize_text(conversation_text)

            # Encode all reference wavs in one batch call
            encoded_list = processor.encode_audios_from_wav(
                wav_list=clone_wavs,
                sampling_rate=sample_rate,
            )

            # Build reference list — one slot per speaker, None where not cloned
            reference_audio_codes = [None] * speaker_count
            for speaker_id, audio_codes in zip(cloned_speakers, encoded_list):
                reference_audio_codes[speaker_id - 1] = audio_codes

            # Encode concatenated prompt wav as the continuation prefix
            concat_prompt_wav = torch.cat(clone_wavs, dim=-1)
            prompt_audio = processor.encode_audios_from_wav(
                wav_list=[concat_prompt_wav],
                sampling_rate=sample_rate,
            )[0]

            user_msg = processor.build_user_message(
                text=conversation_text,
                reference=reference_audio_codes,
            )
            assistant_msg = processor.build_assistant_message(
                audio_codes_list=[prompt_audio],
            )
            conversations = [[user_msg, assistant_msg]]
            mode = "continuation"

        batch = processor(conversations, mode=mode)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
                audio_repetition_penalty=repetition_penalty,
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
