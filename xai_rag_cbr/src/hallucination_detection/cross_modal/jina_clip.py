import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
from typing import List
from .base import BaseCrossModal, CrossModalResult

class JinaClipEvaluator(BaseCrossModal):
    """
    Evaluates cross-modal hallucination using Jina-CLIP-v2, a contrastive 
    model designed for shared text-image embedding spaces.
    """
    def __init__(self, model_id: str = "jinaai/jina-clip-v2"):
        super().__init__()
        self.model_id = model_id
        # Auto-detect device, falling back to CPU for local machine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_id} on {self.device}...")
        self.model = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True, # Jina-CLIP requires trust_remote_code=True because it uses a custom model architecture
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16 # Use float32 on CPU to prevent crash on local machine
        ).to(self.device)
        
        self.model.eval()

    def evaluate(self, image_path: str, texts: List[str]) -> CrossModalResult:
        """
        Extracts embeddings and calculates cosine similarity.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():

            # Jina's custom AutoModel handles tokenization and image preprocessing internally
            img_emb = self.model.encode_image(image)     # Shape: [1, hidden_size]
            txt_embs = self.model.encode_text(texts)     # Shape: [len(texts), hidden_size]

            # Convert Numpy arrays to PyTorch Tensors
            img_emb = torch.as_tensor(img_emb, dtype=torch.float32).to(self.device)
            txt_embs = torch.as_tensor(txt_embs, dtype=torch.float32).to(self.device)

            # Force 2-Dimensional vectors for compatibility
            if img_emb.dim() == 1:
                img_emb = img_emb.unsqueeze(0)
            if txt_embs.dim() == 1:
                txt_embs = txt_embs.unsqueeze(0)
            
            # Normalize the vectors 
            img_emb = F.normalize(img_emb, p=2, dim=1)
            txt_embs = F.normalize(txt_embs, p=2, dim=1)
            
            # Calculate Cosine Similarity via Matrix Multiplication
            similarities = torch.mm(txt_embs, img_emb.transpose(0, 1)).squeeze()
            
            # Handle edge case where there is only 1 text (returns a 0-dim tensor)
            if similarities.dim() == 0:
                scores = [similarities.item()]
            else:
                scores = similarities.tolist()
                
        return CrossModalResult(
            scores=scores,
            method_name="Jina-CLIP-v2",
            metadata={"model_id": self.model_id}
        )