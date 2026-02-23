import torch


def apply_handles(tensor_1d, sample_rate, head_seconds, tail_seconds):
    head_samples = int(head_seconds * sample_rate)
    tail_samples = int(tail_seconds * sample_rate)
    parts = []
    if head_samples > 0:
        parts.append(torch.zeros(head_samples, dtype=tensor_1d.dtype, device=tensor_1d.device))
    parts.append(tensor_1d)
    if tail_samples > 0:
        parts.append(torch.zeros(tail_samples, dtype=tensor_1d.dtype, device=tensor_1d.device))
    return torch.cat(parts) if len(parts) > 1 else tensor_1d


def comfyui_audio_to_moss_tensor(audio_dict):
    waveform = audio_dict["waveform"]   # [B, C, S]
    sr = audio_dict["sample_rate"]
    wav = waveform[0].mean(dim=0)       # [S]
    return wav, sr


def moss_tensor_to_comfyui_audio(tensor_1d, sample_rate=24000):
    return {
        "waveform": tensor_1d.unsqueeze(0).unsqueeze(0),  # [1, 1, S]
        "sample_rate": sample_rate,
    }


def resample_if_needed(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    import torchaudio
    return torchaudio.functional.resample(waveform, orig_sr, target_sr)
