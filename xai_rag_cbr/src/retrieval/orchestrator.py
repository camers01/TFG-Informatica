import time
from typing import List

from src.retrieval.schemas import QueryContext, RetrievedCase
from src.retrieval.manager import CaseBaseManager
from src.retrieval.engines.strict_filter import StrictFilterEngine
from src.retrieval.engines.tabular_scoring import TabularScoringEngine
from src.retrieval.engines.visual_scoring import VisualScoringEngine
from src.retrieval.engines.fusion_scoring import FusionEngine

class RetrievalOrchestrator:
    """
    The central Facade for the Retrieval Module.
    Initializes all engines once and routes queries through the analytical pipeline.
    """
    def __init__(self, data_dir: str):
        
        print("\n[RetrievalOrchestrator] Booting up Retrieval System...")
        start_time = time.time()

        # 1. Initialize the Storage Manager (Loads the CSV)
        self.manager = CaseBaseManager(processed_data_dir=data_dir)

        # 2. Initialize the Engines (Loads MiniLM and ChartGemma weights)
        self.filter_engine = StrictFilterEngine()
        self.tabular_engine = TabularScoringEngine(manager=self.manager)
        self.visual_engine = VisualScoringEngine(manager=self.manager)
        self.fusion_engine = FusionEngine(manager=self.manager)

        boot_time = time.time() - start_time
        print(f"[RetrievalOrchestrator] System Ready. Boot time: {boot_time:.2f} seconds.\n")

    def retrieve(self, query: QueryContext, exclude_id: str = None) -> List[RetrievedCase]:
        """
        Executes the full sequential RAG retrieval pipeline for a user query.
        """
        print(f"[RetrievalOrchestrator] New Query Received")

        # Step 1: Get a copy of the master database from the Manager
        df_current = self.manager.get_full_dataframe()

        # EXTRA step for LEAVE-ONE-OUT
        if exclude_id:
            # Drop the query case from the database so it cannot be retrieved
            df_current = df_current[df_current['id'] != exclude_id]

        # Step 2: Apply Strict Filters with the Filter Engine
        df_current = self.filter_engine.execute(df_current, query)
        if df_current.empty:
            print("[RetrievalOrchestrator] Pipeline Halted: No cases survived strict filtering.")
            return []

        # Step 3: Calculate Tabular Similiratity Scores with the Tabular Engine and recieve top N candidates
        df_current = self.tabular_engine.execute(df_current, query)
        if df_current.empty:
            return []

        # Step 4: Calculate Visual Similarity Scores with the Visual Engine for the top N candidates from the Tabular Engine
        df_current = self.visual_engine.execute(df_current, query)
        if df_current.empty:
            return []

        # Step 5: Apply Late Fusion with the Fusion Engine to combine Tabular and Visual Scores and recieve the final K candidates
        final_results = self.fusion_engine.execute(df_current)

        print(f"[RetrievalOrchestrator] Retrieval Complete. Obtained {len(final_results)} cases.")

        return final_results
    
    def add_new_case(self, case_data: dict, src_image_path: str, raw_images_dir: str) -> str:
        """
        Retrieves the pre-calculated query embeddings from the engines
        and registers the complete new case into the CaseBaseManager.
        """
        if self.manager._df.empty:
            new_id = "000001"
        else:
            new_id = str(self.manager._df['id'].astype(int).max() + 1).zfill(6)
            
        vectors_dict = {}
        
        # 1. Grab Tabular Embeddings (computed during the retrieve step)
        if hasattr(self.tabular_engine, 'latest_domain_emb') and hasattr(self.tabular_engine, 'latest_task_emb'):
            vectors_dict[f"embeddings_domain/{new_id}.npy"] = self.tabular_engine.latest_domain_emb
            vectors_dict[f"embeddings_task/{new_id}.npy"] = self.tabular_engine.latest_task_emb
        else:
            print("[RetrievalOrchestrator] Warning: Tabular embeddings not found in engine cache.")
            
        # 2. Grab Visual Embeddings (computed during the retrieve step)
        if hasattr(self.visual_engine, 'latest_visual_emb'):
            vectors_dict[f"embeddings_visual/{new_id}.npy"] = self.visual_engine.latest_visual_emb
        else:
            print("[RetrievalOrchestrator] Warning: Visual embeddings not found in engine cache.")
            
        # 3. Route to the database storage manager
        return self.manager.add_new_case(
            case_data=case_data,
            src_image_path=src_image_path,
            raw_images_dir=raw_images_dir,
            vectors_dict=vectors_dict
        )