import os
from typing import Dict, List, Optional

from src.retrieval.orchestrator import RetrievalOrchestrator
from src.retrieval.schemas import QueryContext
from src.prompts.prompt_builder import PromptManager
from src.llms.orchestrator import VisionLLMManager
from src.hallucination_detection.manager import HallucinationManager
from src.hallucination_detection.cross_modal.jina_clip import JinaClipEvaluator
from src.hallucination_detection.cross_output.eigen_score import EigenScoreEvaluator
from src.summarizer.manager import SynthesizerManager

class MasterRAGSystem:

    def __init__(self, processed_dir: str, raw_dir: str):
        print("Initializing Master RAG System...")
        self.retrieval = RetrievalOrchestrator(data_dir=processed_dir) 
        self.prompts = PromptManager()
        self.llm_manager = VisionLLMManager()
        self.raw_dir = raw_dir
        # Live pipeline components
        self.jina_evaluator = JinaClipEvaluator()
        self.eigen_evaluator = EigenScoreEvaluator()
        # Hallucination Manager with optimal config: Seq AND Jina(M: 0.35) -> Eigen(S: 0.94)
        self.hallucination_manager = HallucinationManager(
            cross_modal_method=self.jina_evaluator,
            cross_output_method=self.eigen_evaluator,
            aggregation_style="sequential_modal_first",
            cross_modal_style="sequential",
            thresholds={
                "modal_primary": 0.35,   # Medium tier for Jina
                "modal_secondary": 0.00, # Unused
                "output": 0.94           # Strict tier for Eigen
            }
        )
        # Summarizer using Gemma-31B
        self.synthesizer = SynthesizerManager(provider="gemma_31b", is_batch=False)
        
    def process_case(self, case_data: Dict, exclude_id: Optional[str] = None, specific_model: Optional[str] = None) -> Dict:
        """
        Runs the RAG pipeline.
        - If specific_model is provided: Runs a fast, single-model generation (Evaluation Mode).
        - If specific_model is None: Runs the full multi-agent pipeline + Hallucination Check + Summarization + DB Insertion (Live Mode).
        """

        ########################### Common Phase: Retrieval & Prompt Building ###########################
        
        # 1. Map the raw dictionary to the QueryContext defined type
        image_full_path = os.path.join(self.raw_dir, case_data['img_path']) # Get the full image path
        query = QueryContext(
            img_path=image_full_path,
            domain=case_data['domain'],
            ai_task=case_data['ai_task'],
            ai_problem_type=case_data['ai_problem_type'],
            class_value=case_data['class'], 
            library=case_data['library'],
            input_format=case_data['input_format'],
            xai_graph_type=case_data['xai_graph_type'],
            ai_model=case_data['ai_model'],
            explainer=case_data['explainer'],
            attributes=case_data['attributes'],
            scope=case_data['scope'],
            portability=case_data['portability'],
            concurrency=case_data['concurrency']
        )
        
        # 2. Retrieval (passing the exclude_id if using Leave-One-Out evaluation)
        retrieved_objects = self.retrieval.retrieve(query=query, exclude_id=exclude_id)
        retrieved_dicts = [case.metadata for case in retrieved_objects] # Extract the raw metadata dictionaries for the prompt builder
        
        # 3. Build the Few-Shot Prompt using the retrieved cases and the original query
        prompt = self.prompts.build(case_data=case_data, retrieved_cases=retrieved_dicts)

        
        ############################# Branch 1: Single Model (Evaluation Mode) #############################

        if specific_model is not None:

            print("\n[MasterRAG] Triggering Evaluation Mode Single Model Pipeline...")

            result = self.llm_manager.generate_with_specific_model(
                model_name=specific_model,
                image_path=image_full_path,
                prompt=prompt,
                case_id=case_data.get('id', 'NA')
            )

            return result
        
        
        ################################### Branch 2: Full Live Pipeline ###################################

        else: 

            print("\n[MasterRAG] Triggering Full Live Multi-Agent Pipeline...")
            
            # 1. Generate K texts from all models
            vlm_outputs = self.llm_manager.process_live_case(
                image_path=image_full_path, 
                prompt=prompt, 
                case_id=case_data.get('id', 'NA')
            )
            # Isolate model text outputs and filter failed API calls
            raw_texts = []
            for name, text in vlm_outputs.items():
                if name != "case_id":
                    if isinstance(text, str) and not text.startswith("Error:"):
                        raw_texts.append(text.strip())
                    else:
                        print(f"[MasterRAG] Skipped failed or invalid output from {name}: {text}")
            if not raw_texts:
                return {"error": "All Vision LLM generations failed inside the orchestrator."}

            # 2. Hallucination Detection
            print(f"[MasterRAG] Sending {len(raw_texts)} outputs to Hallucination Manager...")
            surviving_texts, report = self.hallucination_manager.filter_outputs(
                image_path=image_full_path, 
                texts=raw_texts
            )
            # Fallback logic: if everything is flagged as hallucinated, retrieve the single highest scorer via Jina-CLIP
            if not surviving_texts:
                print("[MasterRAG] Warning: All generated candidates failed hallucination detection.")
                print("[MasterRAG] Triggering Jina-CLIP fallback...")
                jina_result = self.jina_evaluator.evaluate(image_full_path, raw_texts)
                best_idx = jina_result.scores.index(max(jina_result.scores))
                best_text = raw_texts[best_idx]
                surviving_texts = [best_text]
                print(f"[MasterRAG] Selected candidate index {best_idx} to bypass synthesis failure.")

            # 3. Summarization
            print(f"[MasterRAG] Synthesizing final response from {len(surviving_texts)} candidate(s)...")
            final_insight = self.synthesizer.process(surviving_texts)
            
            # 4. Integrate into Case Base
            print("[MasterRAG] Registering new case into the Case Base...")
            case_registration_data = case_data.copy()
            case_registration_data['solution_insights'] = final_insight
            registered_id = self.retrieval.add_new_case(
                case_data=case_registration_data,
                src_image_path=image_full_path,
                raw_images_dir=os.path.join(self.raw_dir, "xai_images")
            )
            
            return {
                "case_id": registered_id,
                "insight": final_insight,
                "hallucination_report": report,
                "models_used": list(vlm_outputs.keys())
            }