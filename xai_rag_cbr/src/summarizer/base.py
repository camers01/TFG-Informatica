from abc import ABC, abstractmethod
from typing import List

class BaseSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, texts: List[str]) -> str:
        pass