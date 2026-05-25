import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, PaliGemmaModel
from typing import List
from .base import BaseCrossModal, CrossModalResult

class ChartGemmaCosineEvaluator(BaseCrossModal):
    """
    Evaluates cross-modal hallucination by extracting separate vision and language 
    embeddings from ChartGemma and computing their cosine similarity.
    """
    def __init__(self, model_id: str = "ahmed-masry/chartgemma"):
        super().__init__()
        self.model_id = model_id
        # Auto-detect device, falling back to CPU for local machine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_id} (Cosine) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaModel.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16 # Use float32 on CPU to prevent crash on local machine
        ).to(self.device)

        self.model.eval()

    def evaluate(self, image_path: str, texts: List[str]) -> CrossModalResult:
        """
        Extracts image and text embeddings from their respective model towers 
        and calculates cosine similarity.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        scores = []
        
        for text in texts:
            inputs_img = self.processor(images=image, return_tensors="pt").to(self.device)
            inputs_txt = self.processor(text=text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Get Image Embedding
                vision_outputs = self.model.vision_tower(inputs_img.pixel_values)
                img_features = self.model.multi_modal_projector(vision_outputs.last_hidden_state)
                img_emb = img_features.mean(dim=1)
                
                # Get Text Embedding
                text_outputs = self.model.language_model.model(
                    input_ids=inputs_txt.input_ids,
                    attention_mask=inputs_txt.attention_mask
                )
                txt_emb = text_outputs.last_hidden_state.mean(dim=1)
                
                # Normalize the vectors
                img_emb = F.normalize(img_emb, p=2, dim=1)
                txt_emb = F.normalize(txt_emb, p=2, dim=1)

                # Cosine Similarity
                sim = torch.mm(txt_emb, img_emb.transpose(0, 1)).item()
                
                scores.append(sim)
                
        return CrossModalResult(
            scores=scores,
            method_name="ChartGemma-Cosine",
            metadata={"model_id": self.model_id}
        )