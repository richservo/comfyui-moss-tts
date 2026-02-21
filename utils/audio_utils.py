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
