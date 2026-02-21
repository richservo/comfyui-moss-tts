from .nodes.model_loader import MossTTSModelLoader
from .nodes.generate import MossTTSGenerate

NODE_CLASS_MAPPINGS = {
    "MossTTSModelLoader": MossTTSModelLoader,
    "MossTTSGenerate": MossTTSGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSModelLoader": "MOSS-TTS Model Loader",
    "MossTTSGenerate": "MOSS-TTS Generate",
}
