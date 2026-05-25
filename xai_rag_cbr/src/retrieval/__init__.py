"""
Retrieval Module for xAI Case-Based Reasoning.
Exposes only the Orchestrator and the defined Data Types.
"""

from .orchestrator import RetrievalOrchestrator
from .schemas import QueryContext, RetrievedCase
from .config import FUSION_WEIGHT_Z, TOP_N_VISUAL_CANDIDATES, TOP_K_FINAL_RESULTS

__all__ = [
    "RetrievalOrchestrator",
    "QueryContext",
    "RetrievedCase",
]