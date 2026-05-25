import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.schemas import QueryContext
from src.retrieval.manager import CaseBaseManager
from src.retrieval.config import (
    TABULAR_WEIGHTS, 
    TOP_N_VISUAL_CANDIDATES,
    get_model_score,
    get_explainer_score
)
from .base import BaseEngine

class TabularScoringEngine(BaseEngine):
    """
    Computes the tabular similarity between the user's query and the candidate cases. Sorts and truncates to the Top N.
    """
    def __init__(self, manager: CaseBaseManager):
        super().__init__()
        self.manager = manager
        self.log("Loading MiniLM for dynamic text embeddings...")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def _jaccard_sim(self, str1: str, str2: str) -> float:
        """Calculates Jaccard similarity between two comma-separated attribute strings."""
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
            
        set1 = set([x.strip().lower() for x in str1.split(',')])
        set2 = set([x.strip().lower() for x in str2.split(',')])
        
        if not set1 or not set2:
            return 0.0
            
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def _class_sim(self, q_val: str, db_val: str, problem_type: str) -> float:
        """
        Evaluates class difference according to the corrected blueprint.
        - Classification: Exact match required (both Local and Global target specific classes).
        - Regression: "NA" vs "NA" yields 1.0 (Global). Numbers yield relative distance (Local).
        """
        # Clean inputs to handle NaNs and literal "NA" strings consistently
        q_str = str(q_val).strip().upper() if pd.notna(q_val) else "NA"
        db_str = str(db_val).strip().upper() if pd.notna(db_val) else "NA"

        # 1. Classification: Requires an exact match
        if problem_type.lower() == 'classification':
            return 1.0 if q_str == db_str else 0.0
            
        # 2. Regression: Handle Global ("NA") vs Local (Numbers)
        elif problem_type.lower() == 'regression':
            
            # If both are NA, it's a perfect match for Global Regression
            if q_str == "NA" and db_str == "NA":
                return 1.0
                
            # Both are numbers (Local Regression): Calculate relative absolute difference
            try:
                a, b = float(q_val), float(db_val)
                epsilon = 1e-9
                diff = abs(a - b)
                return max(0.0, 1.0 - (diff / (abs(a) + abs(b) + epsilon)))
            except ValueError:
                return 0.0
                
        return 0.0

    def execute(self, df: pd.DataFrame, query: QueryContext) -> pd.DataFrame:

        # 1. Beforehand, we prepare and cache dynamic text vectors for the query
        q_domain_emb = self.text_model.encode(query.domain, normalize_embeddings=True)
        q_task_emb = self.text_model.encode(query.ai_task, normalize_embeddings=True)
        # Save to the engine instance so the Orchestrator can use them later
        self.latest_domain_emb = q_domain_emb
        self.latest_task_emb = q_task_emb

        if df.empty:
            self.log("Received empty DataFrame. Skipping scoring.")
            return df

        self.log(f"Scoring {len(df)} candidate tabular profiles...")

        # 2. Fetch the pre-computed matrices from the manager
        db_domain_matrix = self.manager.get_vectors_batch(df['domain_emb_path'].tolist())
        db_task_matrix = self.manager.get_vectors_batch(df['task_emb_path'].tolist())

        # 3. Vectorized Text Scoring
        domain_raw = np.dot(db_domain_matrix, q_domain_emb)
        task_raw = np.dot(db_task_matrix, q_task_emb)
        # Shift Cosine Similarity output [-1, 1] to strict [0, 1] scale (using min-max normalization for cosine similarity)
        domain_scores = (domain_raw + 1) / 2
        task_scores = (task_raw + 1) / 2

        tabular_scores = []
        
        # 4. Iterate through rows for scalar & hierarchical math
        for idx, (_, row) in enumerate(df.iterrows()):
            
            # AI_task and domain scores (cosine similarity of text embeddings)
            s_domain = domain_scores[idx] * TABULAR_WEIGHTS["domain"]
            s_task = task_scores[idx] * TABULAR_WEIGHTS["ai_task"]
            
            # Attributes score (Jaccard similarity)
            val_attr = self._jaccard_sim(query.attributes, row.get('attributes', ''))
            s_attr = val_attr * TABULAR_WEIGHTS["attributes"]
            
            # Class value score (conditional logic)
            val_class = self._class_sim(
                query.class_value, 
                row.get('class', ''), 
                query.ai_problem_type
            )
            s_class = val_class * TABULAR_WEIGHTS["class_value"]
            
            # AI_model score (hierarchical dictionary)
            val_model = get_model_score(query.ai_model, row.get('ai_model', ''))
            s_model = val_model * TABULAR_WEIGHTS["ai_model"]
            
            # Explainer score (hierarchical dictionary)
            val_explainer = get_explainer_score(query.explainer, row.get('explainer', ''))
            s_explainer = val_explainer * TABULAR_WEIGHTS["explainer"]
            
            # Analytical Family score (categorical match)
            val_family = 1.0 if query.analytical_family == row.get('analytical_family', '') else 0.0
            s_family = val_family * TABULAR_WEIGHTS["analytical_family"]

            # Because weights sum to 1.0, the sum of the weighted scores is the final S_tabular
            total_score = s_domain + s_task + s_attr + s_class + s_model + s_explainer + s_family
            tabular_scores.append(total_score)

        # 5. Append scores and sort descending by tabular_score
        df['tabular_score'] = tabular_scores
        df_sorted = df.sort_values(by='tabular_score', ascending=False)
        
        # 6. Truncate to the Top N (defined in config.py) most similar cases according to tabular_score
        top_n = df_sorted.head(TOP_N_VISUAL_CANDIDATES).copy()
        
        self.log(f"Tabular scoring complete. Passing top {len(top_n)} candidates to visual engine.")
        return top_n