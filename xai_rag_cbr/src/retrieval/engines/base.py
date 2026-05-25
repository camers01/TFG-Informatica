import torch

class BaseEngine:
    """
    Provides shared utilities, device management, and logging 
    for all downstream retrieval engines.
    """
    def __init__(self):
        # Automatically detect if a GPU is available for the math engines
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def log(self, message: str):
        """
        Standardized logging format. Automatically prepends the name 
        of the specific engine running it (e.g., [StrictFilterEngine]).
        """
        print(f"[{self.__class__.__name__}] {message}")