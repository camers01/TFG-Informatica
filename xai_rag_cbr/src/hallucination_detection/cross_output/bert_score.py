import torch
from typing import List
from evaluate import load

from .base import BaseCrossOutput, CrossOutputResult

class BERTScoreEvaluator(BaseCrossOutput):
    """
    Evaluates consensus by calculating how semantically distant each text 
    is from the other texts in the group.
    """
    def __init__(self, model_id: str = "distilbert-base-uncased"):
        super().__init__()
        self.model_id = model_id
        # Auto-detect device, falling back to CPU for local machine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading BERTScore ({self.model_id}) on {self.device}...")
        self.bertscore = load("bertscore")

    def evaluate(self, texts: List[str]) -> CrossOutputResult:
        n = len(texts)
        if n < 2:
            return CrossOutputResult(
                scores=[0.0] * n,
                method_name="BERTScore",
                metadata={"error": "Need at least 2 texts"}
            )

        scores = []
        
        for i in range(n):
            references = [texts[j] for j in range(n) if i != j]
            predictions = [texts[i]] * (n - 1)
            
            results = self.bertscore.compute(
                predictions=predictions, 
                references=references, 
                model_type=self.model_id, 
                device=self.device
            )
            
            avg_f1 = sum(results["f1"]) / len(results["f1"])
            scores.append(avg_f1)

        return CrossOutputResult(
            scores=scores,
            method_name="BERTScore",
            metadata={"model_id": self.model_id}
        )