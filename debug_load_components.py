import torch
from diffusers import AutoPipelineForText2Image, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import gc

MODEL_ID = "stabilityai/sdxl-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Debugging loading for {MODEL_ID} on {DEVICE}")

def report_memory():
    if torch.cuda.is_available():
        print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")

print("\n1. Loading Tokenizer...")
try:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    print("✅ Tokenizer loaded")
except Exception as e:
    print(f"❌ Tokenizer failed: {e}")

print("\n2. Loading Text Encoder...")
try:
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16)
    print("✅ Text Encoder loaded")
    report_memory()
except Exception as e:
    print(f"❌ Text Encoder failed: {e}")

print("\n3. Loading VAE...")
try:
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", variant="fp16", torch_dtype=torch.float16)
    print("✅ VAE loaded")
    report_memory()
except Exception as e:
    print(f"❌ VAE failed: {e}")

print("\n4. Loading UNet (This is usually the biggest part)...")
try:
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", variant="fp16", torch_dtype=torch.float16)
    print("✅ UNet loaded")
    report_memory()
except Exception as e:
    print(f"❌ UNet failed: {e}")

print("\n5. Attempting full pipeline load with low_cpu_mem_usage=True...")
try:
    del tokenizer, text_encoder, vae, unet
    gc.collect()
    torch.cuda.empty_cache()
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        low_cpu_mem_usage=True
    )
    print("✅ Full pipeline loaded successfully!")
except Exception as e:
    print(f"❌ Full pipeline failed: {e}")
