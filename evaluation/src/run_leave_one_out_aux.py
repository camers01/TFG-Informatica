import sys
import os

# Before the import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import shutil
import pandas as pd
from tqdm import tqdm
import torch
import gc

def run_evaluation_loop():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        
    from src.main_rag import MasterRAGSystem
    from src.retrieval.schemas import QueryContext
    from src.llms.orchestrator import VisionLLMManager

    # Paths setup
    KAGGLE_WORKING_DIR = "/kaggle/working"
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
    INPUT_CSV = os.path.join(PROCESSED_DATA_PATH, "case_base.csv")
    PREVIOUS_CHECKPOINT_CSV = os.path.join(PROCESSED_DATA_PATH, "pixtral_rag_insights.csv")
    OUTPUT_CSV = os.path.join(KAGGLE_WORKING_DIR, "pixtral_rag_insights.csv")
    PROMPTS_CSV = os.path.join(KAGGLE_WORKING_DIR, "precomputed_prompts.csv")
    MODEL_TO_USE = "Pixtral-12B"

    # Load master data
    df_master = pd.read_csv(INPUT_CSV, dtype={'id': str})
    df_master['id'] = df_master['id'].astype(str).str.zfill(6)

    # Checkpointing Logic
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_ids = set(df_existing['id'].astype(str).str.zfill(6).tolist())
    elif os.path.exists(PREVIOUS_CHECKPOINT_CSV):
        shutil.copy(PREVIOUS_CHECKPOINT_CSV, OUTPUT_CSV)
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_ids = set(df_existing['id'].astype(str).str.zfill(6).tolist())
    else:
        pd.DataFrame(columns=['id', 'insight']).to_csv(OUTPUT_CSV, index=False)

    # Identify remaining cases
    remaining_cases = df_master[~df_master['id'].isin(processed_ids)]
    print(f"\nRemaining cases to process: {len(remaining_cases)}")

    if len(remaining_cases) == 0:
        print("All cases processed. Exiting.")
        return

    # ==========================================================
    # PHASE 1: PRE-COMPUTE PROMPTS (Uses Retrieval Models)
    # ==========================================================
    print("\n--- PHASE 1: Pre-computing RAG Prompts ---")
    
    # Initialize RAG System but DO NOT load Pixtral
    rag_system = MasterRAGSystem(processed_dir=PROCESSED_DATA_PATH, raw_dir=RAW_DATA_PATH)
    precomputed_data = []

    for index, row in tqdm(remaining_cases.iterrows(), total=len(remaining_cases)):
        case_id = str(row['id']).zfill(6)
        case_dict = row.to_dict()
        image_full_path = os.path.join(RAW_DATA_PATH, case_dict['img_path'])

        # 1. Manually build QueryContext
        query = QueryContext(
            img_path=image_full_path,
            domain=case_dict['domain'],
            ai_task=case_dict['ai_task'],
            ai_problem_type=case_dict['ai_problem_type'],
            class_value=case_dict['class'], 
            library=case_dict['library'],
            input_format=case_dict['input_format'],
            xai_graph_type=case_dict['xai_graph_type'],
            ai_model=case_dict['ai_model'],
            explainer=case_dict['explainer'],
            attributes=case_dict['attributes'],
            scope=case_dict['scope'],
            portability=case_dict['portability'],
            concurrency=case_dict['concurrency']
        )
        
        # 2. Retrieve & Build Prompt
        retrieved_objects = rag_system.retrieval.retrieve(query=query, exclude_id=case_id)
        retrieved_dicts = [case.metadata for case in retrieved_objects]
        prompt = rag_system.prompts.build(case_data=case_dict, retrieved_cases=retrieved_dicts)
        
        precomputed_data.append({
            'id': case_id,
            'img_path': image_full_path,
            'prompt': prompt
        })

    # Save prompts to disk
    df_prompts = pd.DataFrame(precomputed_data)
    df_prompts.to_csv(PROMPTS_CSV, index=False)

    # REMOVE RETRIEVAL MODELS FROM VRAM
    print("Prompts pre-computed. Unloading retrieval models from VRAM...")
    del rag_system
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================
    # PHASE 2: GENERATION (Uses Strictly Pixtral)
    # ==========================================================
    print("\n--- PHASE 2: Pure Generation Pipeline ---")
    
    # Initialize strictly the VisionLLMManager
    llm_manager = VisionLLMManager()
    print(f"Loading {MODEL_TO_USE} into clean memory...")
    llm_manager.load_model(MODEL_TO_USE)

    for index, row in tqdm(df_prompts.iterrows(), total=len(df_prompts)):
        
        case_id = row['id']
        
        # Run generation
        result = llm_manager.generate_with_specific_model(
            model_name=MODEL_TO_USE,
            image_path=row['img_path'],
            prompt=row['prompt'],
            case_id=case_id
        )
        
        insight = result.get('insight', result.get('error', 'UNKNOWN ERROR'))
        
        # Save output to checkpoint
        new_row = pd.DataFrame([{'id': case_id, 'insight': insight}])
        new_row.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

        # Force aggressive memory cleanup
        del result
        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    print("\nGeneration Complete. Unloading model...")
    llm_manager.unload_model(MODEL_TO_USE)

if __name__ == "__main__":
    run_evaluation_loop()