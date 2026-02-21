MODEL_VARIANTS = {
    "MOSS-TTS (Delay 8B)": "OpenMOSS-Team/MOSS-TTS",
    "MOSS-TTS (Local 1.7B)": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "MOSS-TTSD v1.0": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "MOSS-VoiceGenerator": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "MOSS-SoundEffect": "OpenMOSS-Team/MOSS-SoundEffect",
}

DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

TOKENS_PER_SECOND = 12.5

DEFAULT_PARAMS = {
    "OpenMOSS-Team/MOSS-TTS": {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 25,
        "repetition_penalty": 1.0,
    },
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-TTSD-v1.0": {
        "temperature": 1.1,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-VoiceGenerator": {
        "temperature": 1.5,
        "top_p": 0.6,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
    "OpenMOSS-Team/MOSS-SoundEffect": {
        "temperature": 1.5,
        "top_p": 0.6,
        "top_k": 50,
        "repetition_penalty": 1.2,
    },
}

MAX_NEW_TOKENS_DEFAULT = 4096

SAMPLE_RATE = 24000
