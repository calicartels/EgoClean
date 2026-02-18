# VILA

INT8 quantized LongVILA-R1-7B for video understanding on single GPU.

Loads the model at ~9.6GB VRAM (vs ~14GB fp16) using bitsandbytes INT8.
Only the LLM backbone is quantized; vision tower and projector stay in fp16.

## Usage

```
pip install -r requirements-vila.txt
python quantize_longvila.py
```
