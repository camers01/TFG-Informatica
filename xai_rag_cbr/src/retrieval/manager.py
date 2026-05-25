import os
import shutil
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class CaseBaseManager:
    """
    Handles all Data and File I/O operations for the Retrieval Module.
    Loads the Case Base strictly once and lazy-loads/caches .npy vectors on demand.
    """
    def __init__(self, processed_data_dir: str):

        self.data_dir = processed_data_dir
        self.csv_path = os.path.join(self.data_dir, "case_base.csv")
        
        print("Initializing CaseBaseManager...")
        self._df = self._load_dataframe()
        
        # In-memory dictionary cache to prevent redundant hard drive reads.
        # Key: relative path string, Value: loaded numpy array
        self._vector_cache: Dict[str, np.ndarray] = {}

    def _load_dataframe(self) -> pd.DataFrame:
        """
        Loads the CSV and guarantees the integrity of the case IDs.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CRITICAL: Cannot find Case Base at {self.csv_path}")
            
        df = pd.read_csv(self.csv_path, dtype={'id': str})
        
        # Force IDs to be 6-digit strings (prevents Pandas from dropping leading zeros)
        df['id'] = df['id'].astype(str).str.zfill(6)

        return df

    def get_full_dataframe(self) -> pd.DataFrame:
        """
        Returns a fresh copy of the master DataFrame for the StrictFilterEngine.
        Using .copy() prevents downstream engines from accidentally corrupting the master state.
        """
        return self._df.copy()

    def get_case_metadata(self, case_id: str) -> Dict[str, Any]:
        """
        Extracts the entire row for a specific case to be passed to the parent AI.
        """
        # Find the exact row and convert it to a flat dictionary
        case_row = self._df[self._df['id'] == case_id]
        
        if case_row.empty:
            raise ValueError(f"Case ID {case_id} not found in the database.")
            
        return case_row.iloc[0].to_dict()

    def get_vector(self, relative_path: str) -> np.ndarray:
        """
        Lazy-loads a single .npy file. If it was loaded previously, 
        it fetches it instantly from RAM.
        """
        # Return from RAM if we already loaded it
        if relative_path in self._vector_cache:
            return self._vector_cache[relative_path]
            
        # Otherwise, load it from the hard drive
        full_path = os.path.join(self.data_dir, relative_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing vector file: {full_path}")
            
        # Load and cache for next time
        vector = np.load(full_path)
        self._vector_cache[relative_path] = vector
        
        return vector

    def get_vectors_batch(self, relative_paths: List[str]) -> np.ndarray:
        """
        Takes a list of paths and returns a stacked 2D matrix of vectors.
        This allows the engines to perform fast matrix multiplication.
        """
        # List comprehension leveraging the caching mechanism
        vectors = [self.get_vector(path) for path in relative_paths if pd.notna(path)]
        
        # Stack them into a single 2D Numpy array
        return np.vstack(vectors)
    
    def add_new_case(self, case_data: Dict[str, Any], src_image_path: str, raw_images_dir: str, vectors_dict: Dict[str, np.ndarray]) -> str:
        """
        Dynamically registers a brand-new case into the Case Base memory, aligned with case_base.csv format.
        """
    
        # 1. Determine the new unique ID
        if self._df.empty:
            new_id = "000001"
        else:
            new_id = str(self._df['id'].astype(int).max() + 1).zfill(6)
            
        # 2. Copy image file to raw images directory
        os.makedirs(raw_images_dir, exist_ok=True)
        dest_image_name = f"{new_id}.png"
        dest_image_path = os.path.join(raw_images_dir, dest_image_name)
        shutil.copy2(src_image_path, dest_image_path)
        
        # 3. Create the new database row
        new_row = case_data.copy()
        # ID and image path
        new_row['id'] = new_id
        new_row['img_path'] = f"xai_images/{dest_image_name}"
        # Embedding paths
        new_row['domain_emb_path'] = f"embeddings_domain/{new_id}.npy"
        new_row['task_emb_path'] = f"embeddings_task/{new_id}.npy"
        new_row['visual_emb_path'] = f"embeddings_visual/{new_id}.npy"
        # Solution Insight
        new_row['solution_insights'] = case_data.get('solution_insights', case_data.get('insight', ''))
        # We leave blank the evaluation/baseline columns
        new_row['qwen_insight'] = ""
        new_row['pixtral_insight'] = ""
        new_row['idefics_insight'] = ""
        # Remove any temporary keys to prevent pandas from creating new columns in the CSV
        valid_columns = self._df.columns.tolist()
        new_row = {k: v for k, v in new_row.items() if k in valid_columns}
        # Ensure all columns present in the original CSV are in the new row
        for col in valid_columns:
            if col not in new_row:
                new_row[col] = ""

        # 4. Append and write back to disk
        new_df_row = pd.DataFrame([new_row])
        self._df = pd.concat([self._df, new_df_row], ignore_index=True)
        self._df.to_csv(self.csv_path, index=False)
        print(f"[CaseBaseManager] Successfully registered case {new_id} in CSV database.")
        
        # 5. Save to disk and write to RAM cache
        for rel_path, vector in vectors_dict.items():
            full_vector_path = os.path.join(self.data_dir, rel_path)
            os.makedirs(os.path.dirname(full_vector_path), exist_ok=True)
            np.save(full_vector_path, vector)
            self._vector_cache[rel_path] = vector
            
        return new_id