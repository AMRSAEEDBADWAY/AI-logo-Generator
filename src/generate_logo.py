"""
AI Logo Generation - Colab API Client
Connects to SDXL + LogoRedmond LoRA running on Google Colab
"""

import requests
import base64
from PIL import Image
from typing import List, Optional
from io import BytesIO
import os

from config import GENERATION_CONFIG


class LogoGenerator:
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the Logo Generator.
        
        Args:
            api_url: The ngrok URL from your Colab notebook (e.g., "https://xxxx.ngrok.io")
                     If not provided, will try to read from COLAB_API_URL environment variable.
        """
        self.api_url = api_url or os.environ.get("COLAB_API_URL", "")
        self.is_remote = bool(self.api_url)
        
        if self.is_remote:
            print(f"🌐 Using Remote API: {self.api_url}")
            self._verify_connection()
        else:
            print("⚠️ No API URL configured!")
            print("   1. Run SDXL_LogoRedmond_API.ipynb on Google Colab")
            print("   2. Copy the ngrok URL")
            print("   3. Paste it in the Streamlit sidebar")
    
    def _verify_connection(self):
        """Verify the Colab API is reachable"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Connected to Colab API!")
                print(f"   Model: {data.get('model', 'unknown')}")
                return True
        except Exception as e:
            print(f"❌ Cannot connect to Colab API: {e}")
        return False
    
    def set_api_url(self, url: str):
        """Update the API URL at runtime"""
        self.api_url = url.rstrip("/")
        self.is_remote = bool(url)
        if self.is_remote:
            return self._verify_connection()
        return False

    def generate(
        self,
        prompt: str,
        style: str = "modern",
        num_images: int = 4,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        lora_scale: float = 0.9,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate logos via Colab API.
        
        Args:
            prompt: Description of the logo
            style: Style preset (modern, minimal, vintage, playful, corporate, tech, organic)
            num_images: Number of images to generate
            num_steps: Number of inference steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-9 is good)
            lora_scale: Strength of LogoRedmond LoRA (0.8-1.0)
            seed: Random seed for reproducibility
        """
        
        if not self.api_url:
            print("❌ No API URL configured!")
            return []
        
        print("\n" + "=" * 60)
        print(f"🎨 Generating Logos via Colab (SDXL + LogoRedmond)...")
        print(f"Description: {prompt}")
        print(f"Style: {style} | Steps: {num_steps} | LoRA Scale: {lora_scale}")
        print("=" * 60 + "\n")
        
        images = []
        for i in range(num_images):
            print(f"⚡ Image {i+1}/{num_images}...")
            
            try:
                response = requests.post(
                    f"{self.api_url}/generate",
                    json={
                        "prompt": prompt,
                        "style": style,
                        "num_steps": num_steps,
                        "guidance_scale": guidance_scale,
                        "lora_scale": lora_scale,
                        "width": 1024,  # LogoRedmond trained at 1024
                        "height": 1024
                    },
                    timeout=180  # 3 minutes timeout (SDXL is slower)
                )
                
                data = response.json()
                
                if data.get("success"):
                    # Decode base64 image
                    img_data = base64.b64decode(data["image"])
                    img = Image.open(BytesIO(img_data))
                    
                    # Resize to 512 for display if needed
                    img_display = img.copy()
                    img_display.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    
                    images.append(img)  # Keep full resolution
                    print(f"   ✅ Done! (1024x1024)")
                else:
                    print(f"   ❌ Error: {data.get('error', 'Unknown')}")
                    
            except requests.exceptions.Timeout:
                print(f"   ❌ Timeout! Colab might be slow or disconnected.")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print(f"\n🎉 Generated {len(images)} logo(s)!")
        return images

    def generate_with_variations(
        self, 
        base_prompt: str, 
        variations: List[str], 
        style: str = "modern"
    ) -> List[Image.Image]:
        """Generate logos with different variations"""
        all_imgs = []
        for v in variations:
            imgs = self.generate(f"{base_prompt} {v}", style=style, num_images=1)
            all_imgs.extend(imgs)
        return all_imgs


def quick_generate(
    prompt: str, 
    num_images: int = 4, 
    style: str = "modern", 
    api_url: str = None
) -> List[Image.Image]:
    """Quick function to generate logos"""
    return LogoGenerator(api_url=api_url).generate(prompt, style=style, num_images=num_images)


if __name__ == "__main__":
    # Test with your Colab URL
    api_url = input("Enter your Colab ngrok URL: ").strip()
    gen = LogoGenerator(api_url=api_url)
    imgs = gen.generate("modern tech startup", style="tech", num_images=1)
    if imgs:
        imgs[0].save("test_logo.png")
        print("✅ Saved: test_logo.png")
