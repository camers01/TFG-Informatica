from typing import List
from .implementations.gemma_31b import Gemma31BSynthesizer

class SynthesizerManager:
    def __init__(self, provider: str = "gemma_31b", is_batch: bool = False):
        """
        Generic wrapper for the summarization module.
        Args:
            provider: The name of the LLM backend to use.
            is_batch: Whether to enable RPM pacing for large CSV runs.
        """
        if provider == "gemma_31b":
            self.engine = Gemma31BSynthesizer(is_batch_mode=is_batch)
        else:
            raise ValueError(f"Provider {provider} is not implemented.")

    def process(self, vlm_outputs: List[str]) -> str:
        return self.engine.synthesize(vlm_outputs)