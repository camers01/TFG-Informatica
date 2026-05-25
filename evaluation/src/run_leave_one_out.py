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

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Location of the current script
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    # Inject into Python's module search system
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.main_rag import MasterRAGSystem

    # Paths setup (for Kaggle)
    KAGGLE_WORKING_DIR = "/kaggle/working"
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
    IMAGES_PATH = os.path.join(RAW_DATA_PATH, "xai_images")
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
    INPUT_CSV = os.path.join(PROCESSED_DATA_PATH, "case_base.csv")
    PREVIOUS_CHECKPOINT_CSV = os.path.join(PROCESSED_DATA_PATH, "pixtral_rag_insights.csv")
    OUTPUT_CSV = os.path.join(KAGGLE_WORKING_DIR, "pixtral_rag_insights.csv")
    MODEL_TO_USE = "Pixtral-12B"

    # 1. Initialize System
    rag_system = MasterRAGSystem(processed_dir=PROCESSED_DATA_PATH, raw_dir=RAW_DATA_PATH)
    # Load Pixtral into VRAM once at the beginning
    print(f"\nLoading {MODEL_TO_USE} into memory...")
    rag_system.llm_manager.load_model(MODEL_TO_USE)

    # 2. Load the Master Database to iterate over
    df_master = pd.read_csv(INPUT_CSV, dtype={'id': str})
    df_master['id'] = df_master['id'].astype(str).str.zfill(6)

    # ==========================================
    # TEST MODE LIMITER
    # Keep only the first 2 rows. 
    # df_master = df_master.head(2) 
    # print(f"\n[TEST MODE] Running only {len(df_master)} cases.")
    # ==========================================
    
    # 3. Checkpointing (skip already processed cases)
    processed_ids = set()
    # Check if we are resuming in the middle of an active session
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_ids = set(df_existing['id'].astype(str).str.zfill(6).tolist())
        print(f"Resuming active session. {len(processed_ids)} cases already processed.")
    # Check if we are starting a new session but have an older dataset checkpoint
    elif os.path.exists(PREVIOUS_CHECKPOINT_CSV):
        print("Found previous dataset checkpoint. Copying to working directory...")
        shutil.copy(PREVIOUS_CHECKPOINT_CSV, OUTPUT_CSV)
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_ids = set(df_existing['id'].astype(str).str.zfill(6).tolist())
        print(f"Resuming from dataset checkpoint. {len(processed_ids)} cases already processed.")
    # No checkpoint found, start fresh
    else:
        # Create empty CSV with headers if it doesn't exist
        print("No checkpoint found. Starting fresh.")
        pd.DataFrame(columns=['id', 'insight']).to_csv(OUTPUT_CSV, index=False)

    # 4. Leave-One-Out Evaluation Loop
    print("\nStarting Leave-One-Out Generation Pipeline...")
    for index, row in tqdm(df_master.iterrows(), total=len(df_master)):
        
        case_id = str(row['id']).zfill(6)
        
        if case_id in processed_ids:
            continue
            
        case_dict = row.to_dict()
        
        # Execute pipeline (pass the case_id as exclude_id to leave it out of retrieval)
        result = rag_system.process_case(
            case_data=case_dict, 
            exclude_id=case_id, 
            specific_model=MODEL_TO_USE
        )
        
        insight = result.get('insight', result.get('error', 'UNKNOWN ERROR'))
        
        # We save every case immediately to the CSV in Kaggle Working Directory for Checkpointing
        new_row = pd.DataFrame([{'id': case_id, 'insight': insight}])
        new_row.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

        # Force memory cleanup every loop to prevent VRAM fragmentation
        del result
        del case_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Cleanup
    print("\nGeneration Complete. Unloading model...")
    rag_system.llm_manager.unload_model(MODEL_TO_USE)

if __name__ == "__main__":
    run_evaluation_loop()