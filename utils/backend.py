import importlib.metadata
import importlib.util
import torch

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


LOCAL_MODEL_IDS = {
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}


def is_local_model(model_id):
    """Local models use HF GenerationMixin with a custom generation_config.
    Delay models use a custom generate() with audio_* kwargs."""
    return model_id in LOCAL_MODEL_IDS


def run_generation(model, input_ids, attention_mask, model_id, processor,
                   temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    """Unified generate call that handles both Delay and Local model APIs."""
    if is_local_model(model_id):
        from transformers import GenerationConfig

        class _LocalGenerationConfig(GenerationConfig):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.layers = kwargs.get("layers", [{} for _ in range(32)])
                self.do_samples = kwargs.get("do_samples", None)
                self.n_vq_for_inference = kwargs.get("n_vq_for_inference", 32)

        gen_config = _LocalGenerationConfig.from_pretrained(
            model.config._name_or_path or model.config.name_or_path
        )
        gen_config.pad_token_id = processor.tokenizer.pad_token_id
        gen_config.eos_token_id = 151653
        gen_config.max_new_tokens = max_new_tokens
        gen_config.temperature = 1.0
        gen_config.top_p = top_p
        gen_config.top_k = top_k
        gen_config.repetition_penalty = repetition_penalty
        gen_config.use_cache = True
        gen_config.do_sample = False
        gen_config.n_vq_for_inference = model.channels - 1
        gen_config.do_samples = [True] * model.channels

        # Text layer gets different params from audio layers
        text_layer = {
            "repetition_penalty": 1.0,
            "temperature": 1.5,
            "top_p": 1.0,
            "top_k": top_k,
        }
        audio_layer = {
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        gen_config.layers = [text_layer] + [audio_layer] * (model.channels - 1)

        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    else:
        # Delay 8B model — custom generate() with audio_* kwargs
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            audio_temperature=temperature,
            audio_top_p=top_p,
            audio_top_k=top_k,
            audio_repetition_penalty=repetition_penalty,
        )


def _flash_attn_available() -> bool:
    """Check that flash_attn is both importable and has valid package metadata."""
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        importlib.metadata.version("flash_attn")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def resolve_attn_implementation(device, dtype) -> str:
    if (
        str(device).startswith("cuda")
        and _flash_attn_available()
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"
