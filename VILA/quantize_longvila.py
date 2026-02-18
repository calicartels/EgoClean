import sys
import time
import os
import torch
import transformers
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig

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

FPS = 4.0

VIDEO_PATH = "../data/factory_001/rectified_clip_2.mp4"

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user asks a question, "
    "and then you solves it.\n\n"
    "Please first think deeply about the question based on the "
    "given video, and then provide the final answer. The reasoning "
    "process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> "
    "<answer> answer here </answer>.\n\n"
    "Question: {question}"
)

QUESTION = (
    "This is a 7-minute egocentric video from a factory floor at 4 frames per second. "
    "The worker's normal routine is a repetitive press-machine cycle: "
    "pick up material, load it into the press, activate the press, wait, "
    "remove the finished piece, set it aside. This cycle repeats many times. "
    "\n\n"
    "Your task: Watch the ENTIRE video and identify every moment where the "
    "worker BREAKS from this repetitive cycle. Anomalies include: "
    "adjusting or fixing the machine, inspecting a part closely, pausing or hesitating, "
    "walking away, talking to someone, handling a different tool or object, "
    "or any action that is NOT part of the normal pick-load-press-remove cycle. "
    "\n\n"
    "For each anomaly, report:\n"
    "- The approximate time in the video (MM:SS)\n"
    "- What the worker is doing differently\n"
    "- Why it appears to be a break from routine\n"
    "\n"
    "If the entire video shows only the normal cycle with no breaks, say "
    "'No anomalies detected - all frames show standard press cycle.'"
)


def patched_post_config(self_model):
    import bitsandbytes as bnb

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
    import torch

    if not isinstance(images, torch.Tensor) or images.shape[0] <= 8:
        return self_model.mm_projector(self_model.get_vision_tower()(images))

    features_list = []
    chunk_size = 8
    for i in range(0, images.shape[0], chunk_size):
        chunk = images[i : i + chunk_size]
        feat = self_model.get_vision_tower()(chunk)
        feat = self_model.mm_projector(feat)
        features_list.append(feat)

    return torch.cat(features_list, dim=0)


def find_and_patch_vila_class():
    for mod_name, mod in sys.modules.items():
        if "modeling_vila" in mod_name and hasattr(mod, "VILAPretrainedModel"):
            original_post = mod.VILAPretrainedModel.post_config
            mod.VILAPretrainedModel.post_config = patched_post_config

            if hasattr(mod.VILAPretrainedModel, "encode_images"):
                mod.VILAPretrainedModel.encode_images = patched_encode_images
                print("  Patched encode_images (vision chunking)")

            return mod, original_post
    return None, None


def load_quantized():
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    vila_mod, original_post_config = find_and_patch_vila_class()

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

if not ok:
    print("Quantization verification failed.")
    sys.exit(1)

model.config.fps = FPS
print(f"\nfps={FPS} -> ~{int(7 * 60 * FPS)} frames for 7min video")

gen_cfg = model.default_generation_config
gen_cfg.max_new_tokens = 4096
gen_cfg.max_length = 8192
gen_cfg.do_sample = False

prompt = SYSTEM_PROMPT.format(question=QUESTION)

print(f"Running inference on {VIDEO_PATH}...")
torch.cuda.empty_cache()
vram_report()

t0 = time.time()
response = model.generate_content(
    [prompt, {"path": VIDEO_PATH}],
    generation_config=gen_cfg,
)
elapsed = time.time() - t0

print(f"\nInference took {elapsed:.1f}s")
vram_report()

# Full output, no truncation
print("\n" + "=" * 60)
print("FULL RESPONSE:")
print("=" * 60)
print(response)