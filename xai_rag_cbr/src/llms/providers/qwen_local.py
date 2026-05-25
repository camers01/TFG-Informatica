import torch
import gc
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from ..base_provider import VisionLLMProvider

# Selected temperature configuration for Qwen after the experiment
TEMPERATURE_QWEN = 0.01

class QwenProvider(VisionLLMProvider):

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_id = "Qwen/Qwen3-VL-8B-Instruct"

    def get_name(self) -> str:
        return "Qwen-3-VL-8B"

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

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

    def generate(self, image_path: str, prompt: str, temperature: float = TEMPERATURE_QWEN) -> str:

        if self.model is None:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=2048,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
        generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return output_text[0]

    def unload(self) -> None:
        
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()