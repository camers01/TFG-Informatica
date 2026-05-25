import torch
import gc
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from ..base_provider import VisionLLMProvider

# Selected temperature configuration for Idefics after the experiment
TEMPERATURE_IDEFICS = 0.2

class IdeficsProvider(VisionLLMProvider):

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_id = "HuggingFaceM4/Idefics3-8B-Llama3"

    def get_name(self) -> str:
        return "Idefics3-8B"

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
                device_map="auto"
            )

            self.processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, image_path: str, prompt: str, temperature: float = TEMPERATURE_IDEFICS) -> str:

        if self.model is None:
            self.load()

        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to("cuda")

        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=2048,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
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