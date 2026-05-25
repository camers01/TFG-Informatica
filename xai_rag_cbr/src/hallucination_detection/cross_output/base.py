from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class CrossOutputResult:
    """Standardized output for cross-output evaluations (Individual Merit)."""
    scores: List[float] # Individual similarity of each text to the consensus
    method_name: str                  
    metadata: dict = None             

class BaseCrossOutput(ABC):
    """Abstract base class for all Cross-Output hallucination detectors."""

    @abstractmethod
    def evaluate(self, texts: List[str]) -> CrossOutputResult:
        pass