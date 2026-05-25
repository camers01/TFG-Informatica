import torch
import gc
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from ..base_provider import VisionLLMProvider

# Selected temperature configuration for Pixtral after the experiment
TEMPERATURE_PIXTRAL = 0.4

class PixtralProvider(VisionLLMProvider):

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_id = "mistral-community/pixtral-12b"

    def get_name(self) -> str:
        return "Pixtral-12B"

    def load(self) -> None:

        if self.model is None:

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto", 
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image_path: str, prompt: str, temperature: float = TEMPERATURE_PIXTRAL) -> str:

        if self.model is None:
            self.load()

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}]
            }
        ]
        
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs, 
            max_new_tokens=2048,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        generated_text = self.processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        return generated_text

    def unload(self) -> None:
        
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()