import sys
import time
import os
import torch
import transformers
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "Efficient-Large-Model/LongVILA-R1-7B"

QUANT_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# Choice: fps=4.0 on 20min = 4800 frames. Already proven to fit in 24GB INT8.
# Previous run: 321s, 9.6GB VRAM. Plenty of headroom.
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

# Choice: dense timeline extraction over anomaly detection prompt.
# Previous prompt asked model to judge normal vs abnormal — it compressed
# 20 minutes into one observation. LongVILA demos (football, poker, starcraft)
# show it excels at dense play-by-play temporal description.
# We extract the raw timeline, then filter for anomalies ourselves in Python.
QUESTION = (
    "This is a 20-minute egocentric video from a factory floor at 4 frames per second. "
    "The camera is mounted on the worker's head. "
    "\n\n"
    "Describe EVERYTHING the worker does, minute by minute. "
    "For each minute of video (0:00-1:00, 1:00-2:00, 2:00-3:00, etc up to 20:00), "
    "list every distinct action the worker performs with approximate timestamps. "
    "\n\n"
    "Be extremely specific about:\n"
    "- What the hands are doing (picking up, pressing, turning, holding)\n"
    "- What objects are being touched or manipulated\n"
    "- Any pauses, hesitations, or changes in pace\n"
    "- Any interactions with other people\n"
    "- Any time the worker walks away from the main workstation\n"
    "- Body position changes (bending, reaching, stepping back)\n"
    "\n"
    "Do NOT summarize or skip repetitive actions. "
    "Describe each cycle individually even if they look similar. "
    "If the worker repeats the same action 5 times in one minute, describe all 5. "
    "\n\n"
    "Output as a structured timeline with timestamps."
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
print(f"\nfps={FPS} -> ~{int(20 * 60 * FPS)} frames for 20min video")

gen_cfg = model.default_generation_config
# Choice: max_new_tokens=8192 for dense 20-minute play-by-play.
# Previous 4096 limit produced 3 sentences. A minute-by-minute timeline
# of 20 minutes with per-cycle descriptions needs much more room.
# Alternative: 16384 would be safer but generation would take very long.
gen_cfg.max_new_tokens = 8192
gen_cfg.max_length = 16384
gen_cfg.do_sample = False

prompt = SYSTEM_PROMPT.format(question=QUESTION)

print(f"Running inference on {VIDEO_PATH}...")
print(f"Expecting ~5-15 min for vision encoding + generation")
torch.cuda.empty_cache()
vram_report()

t0 = time.time()
response = model.generate_content(
    [prompt, {"path": VIDEO_PATH}],
    generation_config=gen_cfg,
)
elapsed = time.time() - t0

print(f"\nInference took {elapsed:.1f}s ({elapsed/60:.1f} min)")
vram_report()

print("\n" + "=" * 60)
print("FULL RESPONSE:")
print("=" * 60)
print(response)

# Save to file for later analysis
out_path = "../data/factory_001/longvila_timeline_clip2.txt"
with open(out_path, "w") as f:
    f.write(response)
print(f"\nSaved to {out_path}")