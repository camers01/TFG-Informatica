import pandas as pd
from typing import List

from src.retrieval.schemas import RetrievedCase
from src.retrieval.manager import CaseBaseManager
from src.retrieval.config import FUSION_WEIGHT_Z, TOP_K_FINAL_RESULTS
from .base import BaseEngine

class FusionEngine(BaseEngine):
    """
    Applies the Late Fusion weight to combine tabular and visual scores.
    Sorts the final candidates and packages them into strictly typed 
    RetrievedCase objects for the downstream LLM.
    """
    def __init__(self, manager: CaseBaseManager):
        super().__init__()
        self.manager = manager

    def execute(self, df: pd.DataFrame) -> List[RetrievedCase]:
        if df.empty:
            self.log("WARNING: Received empty DataFrame. Returning 0 cases.")
            return []

        self.log(f"Applying Late Fusion (z={FUSION_WEIGHT_Z}) to Top {len(df)} candidates...")

        # 1. The Late Fusion Math
        # S_final = z * S_tabular + (1 - z) * S_visual
        df['final_score'] = (FUSION_WEIGHT_Z * df['tabular_score']) + ((1.0 - FUSION_WEIGHT_Z) * df['visual_score'])

        # 2. Final Sorting
        df_sorted = df.sort_values(by='final_score', ascending=False)
        
        # 3. Truncate to the absolute Top K (defined in config.py)
        top_k_df = df_sorted.head(TOP_K_FINAL_RESULTS)
        
        # 4. Package into RetrievedCase objects
        retrieved_cases: List[RetrievedCase] = []
        
        for _, row in top_k_df.iterrows():

            case_id = str(row['id']).zfill(6)
            
            # Fetch the complete, unfiltered CSV row as a dictionary
            full_metadata = self.manager.get_case_metadata(case_id)
            
            # Instantiate the RetrievedCase Output Object
            case_obj = RetrievedCase(
                case_id=case_id,
                final_score=float(row['final_score']),
                tabular_score=float(row['tabular_score']),
                visual_score=float(row['visual_score']),
                metadata=full_metadata
            )
            
            retrieved_cases.append(case_obj)

        self.log(f"Fusion complete. Returning Top {len(retrieved_cases)} finalized cases.")
        return retrieved_cases