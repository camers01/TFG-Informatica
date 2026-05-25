import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F

from src.retrieval.schemas import QueryContext
from src.retrieval.manager import CaseBaseManager
from .base import BaseEngine

class VisualScoringEngine(BaseEngine):
    """
    Embeds the live query image using the ChartGemma Vision Tower and computes 
    Cosine Similarity against the Top N candidate visual vectors.
    """
    def __init__(self, manager: CaseBaseManager, model_name: str = "ahmed-masry/chartgemma"):
        super().__init__()
        self.manager = manager
        self.log(f"Loading {model_name} Vision Tower onto {self.device.upper()}...")
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def execute(self, df: pd.DataFrame, query: QueryContext) -> pd.DataFrame:
        """
        Calculates S_visual for the incoming Top N dataframe.
        """

        # 1. Load the Query Image
        if not os.path.exists(query.img_path):
            raise FileNotFoundError(f"CRITICAL: Could not find query image at {query.img_path}")
        try:
            query_img = Image.open(query.img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image. Error: {e}")

        # 2. Extract Query Vector
        with torch.no_grad():
            # Prepare pixels (PaliGemma format) - only using image_processor (no text inputs here)
            inputs = self.processor.image_processor(images=query_img, return_tensors="pt").to(self.device)
            # Extract the vector from the vision tower (inputs directly contains 'pixel_values', instead of inputs.pixel_values))
            vision_outputs = self.model.vision_tower(inputs["pixel_values"])
            # Take the mean across the sequence dimension (dim=1) 
            raw_embeddings = vision_outputs.last_hidden_state.mean(dim=1)
            # L2 Normalization
            normalized_emb = F.normalize(raw_embeddings, p=2, dim=1)
            # Move to CPU and convert to Numpy for saving + Flatten to 1D array for dot product math
            q_vector = normalized_emb.cpu().numpy().flatten()
        # Save to the engine instance so the Orchestrator can use it later
        self.latest_visual_emb = q_vector

        if df.empty:
            self.log("Received empty DataFrame. Skipping visual scoring.")
            return df

        self.log(f"Calculating visual similarity for Top {len(df)} candidates...")

        # 3. Fetch pre-computed candidate vectors
        candidate_paths = df['visual_emb_path'].tolist()
        db_matrix = self.manager.get_vectors_batch(candidate_paths)

        # 4. Vectorized Cosine Similarity
        # db_matrix shape is (N, 1152), q_vector shape is (1152,)
        # The dot product returns an array of N similarity scores
        raw_cosine_scores = np.dot(db_matrix, q_vector)

        # 5. Min-Max Scaling: Shift Cosine output [-1, 1] to strict [0, 1] scale
        visual_scores = (raw_cosine_scores + 1) / 2

        # 6. Append to DataFrame and return
        df['visual_score'] = visual_scores
        
        # Clear GPU cache after processing to keep orchestrator stable
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return df.copy()