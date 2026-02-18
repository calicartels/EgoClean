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
    import bitsandbytes as bnb

    # Convert non-quantized LLM layers (embeddings, norms, lm_head) to fp16.
    # We skip Linear4bit modules — calling .to() on them crashes in transformers 4.x.
    for module in self_model.llm.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.compute_dtype = torch.float16
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
    print("  Patched post_config: converted non-quantized LLM layers to fp16")


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
    # Diagnostics: print dtypes of key layers
    print("\nDtype diagnostics:")
    emb = model.llm.get_input_embeddings()
    print(f"  Embedding weight dtype: {emb.weight.dtype}")
    lm_head = model.llm.get_output_embeddings()
    if lm_head is not None:
        print(f"  LM head weight dtype: {lm_head.weight.dtype}")

    # Check first layer norm dtype
    for name, param in model.llm.named_parameters():
        if "norm" in name.lower() or "layernorm" in name.lower():
            print(f"  {name}: {param.dtype}")
            break

    # Test 1: raw LLM forward pass to check logits sanity
    print("\nTest 1: Raw LLM forward pass...")
    test_ids = model.tokenizer("What is 2 + 2? The answer is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.llm(input_ids=test_ids)
    logits = out.logits[0, -1]
    top5 = torch.topk(logits, 5)
    print(f"  Logits stats: min={logits.min():.2f}, max={logits.max():.2f}, "
          f"mean={logits.mean():.2f}, has_nan={logits.isnan().any()}, has_inf={logits.isinf().any()}")
    print(f"  Top-5 next tokens:")
    for val, idx in zip(top5.values, top5.indices):
        token = model.tokenizer.decode([idx])
        print(f"    {idx.item():6d} -> '{token}' (logit={val:.2f})")

    # Test 1.5: LLM.generate with input_ids (bypasses _embed entirely)
    print("\nTest 1.5: LLM.generate with input_ids directly...")
    gen_out = model.llm.generate(input_ids=test_ids, max_new_tokens=30, do_sample=False)
    decoded = model.tokenizer.decode(gen_out[0], skip_special_tokens=True)
    print(f"  Direct LLM output: {decoded[:200]}")

    # Test 1.6: LLM.generate with inputs_embeds (same path as generate_content)
    print("\nTest 1.6: LLM.generate with inputs_embeds...")
    embed_layer = model.llm.get_input_embeddings()
    test_embeds = embed_layer(test_ids).to(torch.float16)
    print(f"  Embed stats: dtype={test_embeds.dtype}, "
          f"min={test_embeds.min():.4f}, max={test_embeds.max():.4f}, "
          f"has_nan={test_embeds.isnan().any()}")
    gen_out2 = model.llm.generate(inputs_embeds=test_embeds, max_new_tokens=30, do_sample=False)
    decoded2 = model.tokenizer.decode(gen_out2[0], skip_special_tokens=True)
    print(f"  Embeds-based output: {decoded2[:200]}")

    # Test 2: generate_content with model's own default config
    print("\nTest 2: generate_content with default config...")
    vram_snapshot("Before inference")
    gen_cfg = model.default_generation_config
    gen_cfg.max_new_tokens = 50
    gen_cfg.max_length = 200
    gen_cfg.do_sample = False
    gen_cfg.repetition_penalty = 1.2
    response = model.generate_content(["What is 2 + 2?"], generation_config=gen_cfg)
    vram_snapshot("After inference")
    print(f"Response: {response[:200]}")

vram_snapshot("Final")

# To test with video:
# model.config.fps = 0.5
# response = model.generate_content(["Describe this video.", {"path": "video.mp4"}])
# print(response)
