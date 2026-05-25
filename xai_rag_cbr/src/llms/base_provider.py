from abc import ABC, abstractmethod

class VisionLLMProvider(ABC):
    
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads the model weights into GPU memory. 
        If it's an API (Gemini), this can just be `pass`.
        """
        pass

    @abstractmethod
    def generate(self, image_path: str, prompt: str, temperature: float = 0.2) -> str:
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Deletes the model from memory and clears the CUDA cache.
        """
        pass