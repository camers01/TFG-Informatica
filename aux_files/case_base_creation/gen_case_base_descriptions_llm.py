import csv
import os
import logging
import sys
from llms import VisionLLMManager
from llms.config import is_kaggle_environment
from prompts import PromptManager

# Configure logging to track progress in console/Kaggle
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Acess the raw data directory 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INPUT_CSV = os.path.join(RAW_DATA_DIR, "xai_case_base.csv")

# Temperature for LLM generation
# GENERATION_TEMPERATURE = 0.2    (!) In order to use each providers temperature

# To ensure Kaggle writes to the working directory, not the read-only input directory
if is_kaggle_environment():
    CHECKPOINT_CSV = "/kaggle/working/raw_case_outputs.csv"
else:
    CHECKPOINT_CSV = os.path.join(RAW_DATA_DIR, "raw_case_outputs.csv")

def get_completed_cases(checkpoint_file: str) -> set:
    """Reads the checkpoint CSV and returns a set of (case_id, model_name) that are already done."""

    completed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('error'):
                    completed.add((row['case_id'], row['model']))
    return completed

def run_batch_generation():

    logger.info("Initializing Prompt Manager...")
    prompt_manager = PromptManager(version="default")

    logger.info("Initializing LLM Manager...")
    llm_manager = VisionLLMManager()
    available_models = llm_manager.get_available_models()
    
    completed_tasks = get_completed_cases(CHECKPOINT_CSV)
    logger.info(f"Found {len(completed_tasks)} previously completed generations.")

    if not os.path.exists(INPUT_CSV):
        logger.critical(f"Input CSV not found at {INPUT_CSV}. Check your directory structure.")
        sys.exit(1)

    with open(INPUT_CSV, mode='r', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))

    file_exists = os.path.exists(CHECKPOINT_CSV)

    with open(CHECKPOINT_CSV, mode='a', newline='', encoding='utf-8') as f:

        fieldnames = ['case_id', 'model', 'insight', 'error']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for model_name in available_models:
            logger.info(f"--- Starting generation for model: {model_name} ---")
            
            try:
                llm_manager.load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}. Skipping this model.")
                continue

            for case in cases:
                case_id = case['id']
                
                if (case_id, model_name) in completed_tasks:
                    continue
                
                logger.info(f"Processing case {case_id} with {model_name}...")
                
                # Resolves 'xai_images/000000.png' to '.../data/raw/xai_images/000000.png'
                image_path = os.path.join(RAW_DATA_DIR, case['img_path'])

                prompt = prompt_manager.build(case_data=case)
                
                result = llm_manager.generate_with_specific_model(
                    model_name=model_name,
                    image_path=image_path,
                    prompt=prompt,
                    case_id=case_id
                    # temperature=GENERATION_TEMPERATURE    (!) In order to use each providers temperature
                )
                
                row_to_write = {
                    'case_id': case_id,
                    'model': model_name,
                    'insight': result.get('insight', ''),
                    'error': result.get('error', '')
                }
                
                writer.writerow(row_to_write)
                f.flush() 
                
                if row_to_write['error']:
                    logger.critical(f"FATAL ERROR on case {case_id} with {model_name}: {row_to_write['error']}")
                    logger.critical("Halting script to prevent further failure.")
                    llm_manager.unload_model(model_name)
                    sys.exit(1)

            llm_manager.unload_model(model_name)
            logger.info(f"--- Finished all cases for {model_name} ---")

if __name__ == "__main__":
    run_batch_generation()