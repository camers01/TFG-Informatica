import logging
import re
from typing import Dict, List, Optional
from .config import is_kaggle_environment
# from .providers.gemini_api import GeminiProvider
from .providers.qwen_local import QwenProvider
# from .providers.llama_local import LlamaProvider
# from .providers.phi_local import PhiProvider
from .providers.pixtral_local import PixtralProvider
from .providers.idefics_local import IdeficsProvider

logger = logging.getLogger(__name__)

class VisionLLMManager:

    def __init__(self):
        
        self.providers = {} # Store providers in a dictionary for easy targeting by name
        
        # DELETED FOR API USAGE LIMITS
        # gemini = GeminiProvider()
        # self.providers[gemini.get_name()] = gemini
        
        if is_kaggle_environment():

            # We will only load local models if in a Kaggle environment
            qwen = QwenProvider()
            # llama = LlamaProvider()  # DELETED FOR REGION RESTRICTIONS
            # phi = PhiProvider()      # DELETED FOR CODE INCOMPATIBILITY
            pixtral = PixtralProvider()
            idefics = IdeficsProvider()

            self.providers[qwen.get_name()] = qwen
            # self.providers[llama.get_name()] = llama
            # self.providers[phi.get_name()] = phi
            self.providers[pixtral.get_name()] = pixtral
            self.providers[idefics.get_name()] = idefics

    def get_available_models(self) -> List[str]:
        """Returns a list of model names currently available."""
        return list(self.providers.keys())

    ####### METHODS FOR BATCH GENERATION (INITIAL CASE BASE PIPELINE) #######

    def load_model(self, model_name: str) -> None:
        """Explicitly loads a specific model into VRAM."""
        if model_name in self.providers:
            logger.info(f"Loading {model_name} into memory...")
            self.providers[model_name].load()
        else:
            raise ValueError(f"Model {model_name} not found.")

    def unload_model(self, model_name: str) -> None:
        """Explicitly unloads a specific model from VRAM."""
        if model_name in self.providers and is_kaggle_environment():
            logger.info(f"Unloading {model_name} from memory...")
            self.providers[model_name].unload()

    def generate_with_specific_model(self, model_name: str, image_path: str, prompt: str, case_id: str = None) -> Dict[str, str]: #, temperature: float = 0.2) -> Dict[str, str]:    (!) In order to use each providers temperature     
        """
        USE CASE: CSV Case Base Setup.
        Generates an insight for ONE case using ONE specific model that is already loaded.
        """
        if model_name not in self.providers:
            return {"error": f"Model {model_name} not found."}
            
        provider = self.providers[model_name]
        
        try:

            # Get the raw text output from the provider
            insight = provider.generate(image_path, prompt) #, temperature=temperature)    (!) In order to use each providers temperature

            return {
                "case_id": case_id,
                "model": model_name,
                "insight": insight.strip()
            }
        
        except Exception as e:
            return {
                "case_id": case_id,
                "model": model_name,
                "error": str(e)
            }

    ####### METHOD FOR LIVE SYSTEM (FINAL USER PIPELINE) #######

    def process_live_case(self, image_path: str, prompt: str, case_id: str = None) -> Dict[str, str]: #, temperature: float = 0.2) -> Dict[str, str]:    (!) In order to use each providers temperature 
        """
        USE CASE: Final RAG User System.
        Automatically cycles through all available models for a single input,
        loading and unloading them on the fly to fit within VRAM constraints.
        """
        results = {"case_id": case_id} if case_id else {}
        
        for model_name, provider in self.providers.items():
            try:
                provider.load()
                results[model_name] = provider.generate(image_path, prompt) #, temperature=temperature)    (!) In order to use each providers temperature
            except Exception as e:
                results[model_name] = f"Error: {str(e)}"
            finally:
                if is_kaggle_environment():
                    provider.unload()
                    
        return results