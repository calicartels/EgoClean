import sys
import time
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig

MODEL_ID = "Efficient-Large-Model/LongVILA-R1-7B"
QUANT_TYPE = "nf4"
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type=QUANT_TYPE,
    bnb_4bit_use_double_quant=False,
)


def vram_snapshot(label):
    if not torch.cuda.is_available():
        print(f"[{label}] No CUDA available")
        return None
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9
    res = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - alloc
    print(f"[{label}] VRAM: {alloc:.2f}/{total:.1f}GB allocated, "
          f"{res:.2f}GB reserved, {peak:.2f}GB peak, {free:.1f}GB free")
    return alloc, res


def patched_post_config(self_model):
    # Only convert vision_tower and mm_projector — LLM stays quantized
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
    print("  Patched post_config: skipped .to(float16) on quantized LLM")


def find_and_patch_vila_class():
    for mod_name, mod in sys.modules.items():
        if "modeling_vila" in mod_name and hasattr(mod, "VILAPretrainedModel"):
            original = mod.VILAPretrainedModel.post_config
            mod.VILAPretrainedModel.post_config = patched_post_config
            return mod, original
    return None, None


def load_quantized():
    vram_snapshot("Before loading")

    # Phase 1: Load config to trigger remote code download
    print("Downloading remote code...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Phase 2: Patch post_config on the now-imported VILAPretrainedModel
    vila_mod, original_post_config = find_and_patch_vila_class()
    if vila_mod:
        print("  Patched VILAPretrainedModel.post_config")
    else:
        print("  WARNING: could not find VILAPretrainedModel to patch")

    # Phase 3: Patch AutoModelForCausalLM.from_pretrained to inject quantization
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        if "quantization_config" not in kwargs:
            kwargs["quantization_config"] = QUANT_CONFIG
            print("  Injected INT4 quantization into LLM loading")
        return original_from_pretrained(*args, **kwargs)

    pbar = tqdm(total=1, desc="Loading LongVILA-R1-7B", unit="model")
    transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    try:
        t0 = time.time()
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        elapsed = time.time() - t0
        pbar.update(1)
        pbar.set_postfix_str(f"done in {elapsed:.1f}s")
        pbar.close()
    except Exception as e:
        pbar.close()
        print(f"Loading failed: {e}")
        raise
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original_from_pretrained
        if vila_mod and original_post_config:
            vila_mod.VILAPretrainedModel.post_config = original_post_config

    print(f"Model loaded in {elapsed:.1f}s")
    vram_snapshot("After loading")
    return model


def verify(model):
    import bitsandbytes as bnb

    total = sum(p.numel() for p in tqdm(list(model.parameters()),
                                         desc="Counting parameters", unit="param"))
    print(f"Total params: {total / 1e9:.2f}B")

    modules = list(model.llm.named_modules())
    n_quant, n_regular = 0, 0
    for _, m in tqdm(modules, desc="Checking layer types", unit="module"):
        if isinstance(m, bnb.nn.Linear4bit):
            n_quant += 1
        elif isinstance(m, torch.nn.Linear):
            n_regular += 1

    print(f"LLM layers: {n_quant} quantized (Linear4bit), {n_regular} regular (Linear)")

    if n_quant == 0:
        print("WARNING: no quantized layers found, quantization may have failed")
        return False
    return True


def print_vram_report(before, after):
    if before is None or after is None:
        return
    alloc_before, _ = before
    alloc_after, _ = after
    delta = alloc_after - alloc_before
    print(f"\nVRAM Report:")
    print(f"  Before load: {alloc_before:.2f}GB")
    print(f"  After load:  {alloc_after:.2f}GB")
    print(f"  Delta:       {delta:.2f}GB (model footprint)")
    print(f"  Expected ~4-5GB for INT4, ~14GB for fp16")


print("=" * 50)
print("LongVILA-R1-7B INT4 Quantization")
print("=" * 50)

vram_before = vram_snapshot("Baseline")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

model = load_quantized()
vram_after = vram_snapshot("Post-load")

ok = verify(model)
print_vram_report(vram_before, vram_after)

if ok:
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(max_new_tokens=100, do_sample=False, repetition_penalty=1.2)

    print("\nTesting text-only inference...")
    vram_snapshot("Before inference")
    response = model.generate_content(["What is 2 + 2?"], generation_config=gen_cfg)
    vram_snapshot("After inference")
    print(f"Response: {response[:200]}")

vram_snapshot("Final")

# To test with video:
# model.config.fps = 0.5
# response = model.generate_content(["Describe this video.", {"path": "video.mp4"}])
# print(response)
