import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import List
from .base import BaseCrossModal, CrossModalResult

class ChartGemmaLogitEvaluator(BaseCrossModal):
    """
    Evaluates cross-modal hallucination using ChartGemma via visual entailment, 
    prompting the model to validate the text against the chart.
    """
    def __init__(self, model_id: str = "ahmed-masry/chartgemma"):
        super().__init__()
        self.model_id = model_id
        # Auto-detect device, falling back to CPU for local machine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_id} (Logit) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16 # Use float32 on CPU to prevent crash on local machine
        ).to(self.device)

        self.model.eval()

    def evaluate(self, image_path: str, texts: List[str]) -> CrossModalResult:
        """
        Calculates the logit probability of the model predicting 'Yes' when asked 
        if the chart supports the provided statement.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        scores = []
        
        # Token IDs for 'Yes' and 'No' (We look at both to calculate relative probability)
        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]

        for text in texts:
            
            prompt = f"<image>Does the chart support the following statement? Statement: {text}\nAnswer 'Yes' or 'No'."
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get the logits for the very next predicted token
            next_token_logits = outputs.logits[0, -1, :]
            
            yes_logit = next_token_logits[yes_token_id].item()
            no_logit = next_token_logits[no_token_id].item()
            
            # Apply softmax to get probability between 0 and 1
            prob_tensor = torch.nn.functional.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
            yes_probability = prob_tensor[0].item()
            
            scores.append(yes_probability)
            
        return CrossModalResult(
            scores=scores,
            method_name="ChartGemma-Logit",
            metadata={"model_id": self.model_id}
        )