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
