from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class CrossModalResult:
    """Standardized output for cross-modal evaluations."""
    scores: List[float]          # e.g., [0.85, 0.92, 0.21] for the 3 inputs
    method_name: str             # e.g., "ChartGemma"
    metadata: dict = None        # Any extra debug info you want to log

class BaseCrossModal(ABC):
    """Abstract base class for all Cross-Modal hallucination detectors."""
    
    def __init__(self, **kwargs):
        # This allows child classes to accept specific model paths or configs
        pass

    @abstractmethod
    def evaluate(self, image_path: str, texts: List[str]) -> CrossModalResult:
        """
        Compares the graph image against the generated texts.
        
        Args:
            image_path: Path to the SHAP/LIME graph screenshot.
            texts: List of candidate outputs (Gemini + Kaggle models).
            
        Returns:
            CrossModalResult: Contains the grounding scores for each text.
        """
        pass