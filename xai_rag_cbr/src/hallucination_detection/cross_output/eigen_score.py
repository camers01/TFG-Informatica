import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer, util
from .base import BaseCrossOutput, CrossOutputResult

class EigenScoreEvaluator(BaseCrossOutput):
    """
    Evaluates consensus by calculating the Cosine Similarity of each text 
    compared to the mathematical center (centroid) of the group.
    """
    def __init__(self, model_id: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_id = model_id
        # Auto-detect device, falling back to CPU for local machine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_id} on {self.device}...")
        self.model = SentenceTransformer(self.model_id, device=self.device)

    def evaluate(self, texts: List[str]) -> CrossOutputResult:
        if len(texts) < 2:
            return CrossOutputResult(
                scores=[0.0] * len(texts),
                method_name="EigenScore",
                metadata={"error": "Need at least 2 texts"}
            )

        # 1. Embed texts (returns a PyTorch tensor)
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        
        # 2. Calculate the Centroid (average embedding)
        centroid = torch.mean(embeddings, dim=0)
        
        # 3. Calculate Cosine Similarity against the centroid
        # util.cos_sim returns a matrix, we just need the 1D list of scores
        similarities = util.cos_sim(embeddings, centroid).squeeze().tolist()
        
        # 4. Optional: We calculate the global EigenScore (variance) just to log it in metadata
        embeddings_np = embeddings.cpu().numpy()
        centroid_np = centroid.cpu().numpy()
        centered_embeddings = embeddings_np - centroid_np
        cov_matrix = np.dot(centered_embeddings, centered_embeddings.T) / (len(texts) - 1)
        global_variance = float(np.max(np.real(np.linalg.eigvals(cov_matrix))))

        return CrossOutputResult(
            scores=similarities,
            method_name="EigenScore",
            metadata={"model_id": self.model_id, "global_variance": global_variance}
        )