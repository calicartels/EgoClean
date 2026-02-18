import sys
import time
import os
import torch
import transformers
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig

# Help with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "Efficient-Large-Model/LongVILA-R1-7B"

# Choice: INT8 over NF4 (INT4).
# Qwen2 backbone has known KV cache corruption with bnb 4-bit
# that produces garbage during autoregressive generation.
# INT8 uses ~9.6GB vs ~5GB, but generation works correctly.
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)


def patched_post_config(self_model):
    import bitsandbytes as bnb

    # Convert non-quantized LLM layers (embeddings, norms, lm_head) to fp16.
    # Skip Int8 linear modules — .to() on them crashes in transformers 4.x.
    for module in self_model.llm.modules():
        if isinstance(module, bnb.nn.Linear8bitLt):
            continue
        for param in module.parameters(recurse=False):
            param.data = param.data.to(torch.float16)

    self_model.mm_projector = self_model.mm_projector.to(torch.float16)
    self_model.vision_tower = self_model.vision_tower.to(torch.float16)

    self_model.training = self_model.llm.training
    if self_model.training:
        self_model.train()
    else:
        self_model.eval()
    if getattr(self_model.config, "llm_cfg", None) is None:
        self_model.config.llm_cfg = self_model.llm.config
    if getattr(self_model.config, "vision_tower_cfg", None) is None:
        self_model.config.vision_tower_cfg = self_model.vision_tower.config
    if getattr(self_model.config, "mm_projector_cfg", None) is None:
        self_model.config.mm_projector_cfg = self_model.mm_projector.config


def patched_encode_images(self_model, images):
    # Vision Encoder Chunking Patch
    # SigLIP OOMs if we process 600 frames at once.
    # We chunk the input tensor (Batch, C, H, W) and process in small batches.
    import torch
    
    # If not a tensor or small enough, use original logic
    if not isinstance(images, torch.Tensor) or images.shape[0] <= 8:
        return self_model.mm_projector(self_model.get_vision_tower()(images))

    # Chunking logic
    features_list = []
    chunk_size = 8
    for i in range(0, images.shape[0], chunk_size):
        chunk = images[i : i + chunk_size]
        # Vision tower + projector for each chunk
        feat = self_model.get_vision_tower()(chunk)
        feat = self_model.mm_projector(feat)
        features_list.append(feat)
    
    return torch.cat(features_list, dim=0)


def find_and_patch_vila_class():
    for mod_name, mod in sys.modules.items():
        if "modeling_vila" in mod_name and hasattr(mod, "VILAPretrainedModel"):
            # Patch post_config
            original_post = mod.VILAPretrainedModel.post_config
            mod.VILAPretrainedModel.post_config = patched_post_config
            
            # Patch encode_images for chunking
            # (Backup original if needed, but we just replace it)
            if hasattr(mod.VILAPretrainedModel, "encode_images"):
                original_encode = mod.VILAPretrainedModel.encode_images
                mod.VILAPretrainedModel.encode_images = patched_encode_images
                print("  Patched VILAPretrainedModel.encode_images (vision chunking)")
            
            return mod, original_post
    return None, None


def load_quantized():
    # Phase 1: load config to trigger remote code download
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Phase 2: patch post_config on the now-imported VILAPretrainedModel
    vila_mod, original_post_config = find_and_patch_vila_class()

    # Phase 3: patch AutoModelForCausalLM.from_pretrained to inject quantization
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        if "quantization_config" not in kwargs:
            kwargs["quantization_config"] = QUANT_CONFIG
        return original_from_pretrained(*args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    elapsed = time.time() - t0

    transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
    if vila_mod and original_post_config:
        vila_mod.VILAPretrainedModel.post_config = original_post_config

    print(f"Model loaded in {elapsed:.1f}s")

    # Instance-level patch to ensure chunking is used
    # Class-level patching can be tricky with inheritance/shadowing
    import types
    if hasattr(model, "encode_images"):
        model.encode_images = types.MethodType(patched_encode_images, model)
        print("Patched model.encode_images (vision chunking enabled)")

    return model


def verify(model):
    import bitsandbytes as bnb

    total = sum(p.numel() for p in model.parameters())
    modules = list(model.llm.named_modules())
    n_quant = sum(1 for _, m in modules if isinstance(m, bnb.nn.Linear8bitLt))
    n_regular = sum(1 for _, m in modules if isinstance(m, torch.nn.Linear))

    print(f"Params: {total / 1e9:.2f}B | "
          f"LLM: {n_quant} quantized (Linear8bit), {n_regular} regular (Linear)")
    return n_quant > 0


def vram_report():
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - alloc
    print(f"VRAM: {alloc:.2f}/{total:.1f}GB allocated, {free:.1f}GB free")


model = load_quantized()
ok = verify(model)
vram_report()

if ok:
    gen_cfg = model.default_generation_config
    gen_cfg.max_new_tokens = 2048
    gen_cfg.max_length = 4096
    gen_cfg.do_sample = False
    # Video inference on rectified egocentric clip
    # Choice: fps=0.5 (~600 frames) works now thanks to vision chunking patch!
    # Without chunking, even 30 frames (fps=0.025) caused OOM.
    VIDEO_PATH = "../data/factory_001/rectified_clip_2.mp4"
    model.config.fps = 0.5
    
    print(f"\nRunning video inference on {VIDEO_PATH}...")
    torch.cuda.empty_cache()
    vram_report()
    
    response = model.generate_content(
        [
            f"Watch this egocentric video carefully. The video is sampled at {model.config.fps} frames per second. "
            "List the timestamps and frame numbers where the person performs a non-repetitive action. "
            "Format: [Frame X - Time MM:SS] Description.",
            {"path": VIDEO_PATH},
        ],
        generation_config=gen_cfg,
    )
    print(f"\n> Non-repetitive actions:\n{response[:1000]}")
    vram_report()
