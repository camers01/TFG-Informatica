from typing import List, Tuple, Dict, Optional
from .cross_modal.base import BaseCrossModal
from .cross_output.base import BaseCrossOutput

class HallucinationManager:
    """The central orchestrator for hallucination detection in the xAI RAG pipeline."""
    
    def __init__(
        self, 
        cross_modal_method: Optional[BaseCrossModal] = None,
        secondary_cross_modal_method: Optional[BaseCrossModal] = None,
        cross_output_method: Optional[BaseCrossOutput] = None,
        aggregation_style: str = "sequential_modal_first",
        cross_modal_style: str = "fallback",
        thresholds: Dict[str, float] = {"modal_primary": 0.85, "modal_secondary": 0.25, "output": 0.5},
        weights: Dict[str, float] = {"modal": 0.5, "output": 0.5}
    ):
        self.cross_modal = cross_modal_method
        self.secondary_cross_modal = secondary_cross_modal_method
        self.cross_output = cross_output_method
        self.aggregation_style = aggregation_style
        self.cross_modal_style = cross_modal_style
        self.thresholds = thresholds                         
        self.weights = weights

    def filter_outputs(self, image_path: str, texts: List[str]) -> Tuple[List[str], Dict]:
        """The main entry point called by the CBR-RAG system."""

        report = {"original_count": len(texts), "dropped_by": []}
        surviving_texts = texts.copy()

        # Routing Logic
        if self.aggregation_style == "single":
            surviving_texts = self._run_single(image_path, surviving_texts, report)
        elif self.aggregation_style == "sequential_modal_first":
            surviving_texts = self._run_sequential_modal_first(image_path, surviving_texts, report)
        elif self.aggregation_style == "sequential_output_first":
            surviving_texts = self._run_sequential_output_first(image_path, surviving_texts, report)
        elif self.aggregation_style == "or_modal_first":          
            surviving_texts = self._run_or_modal_first(image_path, surviving_texts, report)
        elif self.aggregation_style == "or_output_first":         
            surviving_texts = self._run_or_output_first(image_path, surviving_texts, report)
        elif self.aggregation_style == "weighted":
            surviving_texts = self._run_weighted(image_path, surviving_texts, report)
            
        report["final_count"] = len(surviving_texts)
        return surviving_texts, report

    def _run_single(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Runs whichever method is provided."""
        if self.cross_modal:
            return self._run_cross_modal(image_path, texts, report)
            
        if self.cross_output:
            result = self.cross_output.evaluate(texts)
            return self._filter_by_output(texts, result, report)
            
        return texts

    def _run_sequential_modal_first(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Runs Visual cascade check first, then Consensus check on the survivors."""
        surviving = texts.copy()
        
        if self.cross_modal:
            surviving = self._run_cross_modal(image_path, surviving, report)
            
        if self.cross_output and len(surviving) > 1:
            output_result = self.cross_output.evaluate(surviving)
            surviving = self._filter_by_output(surviving, output_result, report)
            
        return surviving

    def _run_sequential_output_first(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Runs Consensus check first, then Visual cascade check on the survivors."""
        surviving = texts.copy()
        
        if self.cross_output and len(surviving) > 1:
            output_result = self.cross_output.evaluate(surviving)
            surviving = self._filter_by_output(surviving, output_result, report)
            
        if self.cross_modal and len(surviving) > 0:
            surviving = self._run_cross_modal(image_path, surviving, report)
            
        return surviving

    def _run_cross_modal(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Handles multiple visual evaluators using either Fallback (OR) or Sequential (AND) logic."""
        if not self.cross_modal:
            return texts

        # 1. Evaluate with Primary Method
        primary_result = self.cross_modal.evaluate(image_path, texts)
        
        # METHOD A: FALLBACK (OR)

        if self.cross_modal_style == "fallback":

            surviving_indices = set()
            needs_secondary_texts = []
            needs_secondary_indices = []

            for idx, score in enumerate(primary_result.scores):
                if score >= self.thresholds["modal_primary"]:
                    surviving_indices.add(idx)
                else:
                    needs_secondary_texts.append(texts[idx])
                    needs_secondary_indices.append(idx)

            if needs_secondary_texts and self.secondary_cross_modal:
                secondary_result = self.secondary_cross_modal.evaluate(image_path, needs_secondary_texts)
                for f_idx, score in enumerate(secondary_result.scores):
                    original_idx = needs_secondary_indices[f_idx]
                    if score >= self.thresholds["modal_secondary"]:
                        surviving_indices.add(original_idx)
                    else:
                        report["dropped_by"].append({
                            "text_index": original_idx, 
                            "reason": f"failed_fallback (Pri: {primary_result.scores[original_idx]:.2f}, Sec: {score:.2f})", 
                            "score": score
                        })
            else:
                for f_idx, text in enumerate(needs_secondary_texts):
                    original_idx = needs_secondary_indices[f_idx]
                    report["dropped_by"].append({
                        "text_index": original_idx, 
                        "reason": f"{primary_result.method_name}_low_grounding", 
                        "score": primary_result.scores[original_idx]
                    })

            return [texts[i] for i in range(len(texts)) if i in surviving_indices]


        # METHOD B: SEQUENTIAL (AND)

        elif self.cross_modal_style == "sequential":

            surviving_texts = []
            surviving_indices = []
            
            for idx, score in enumerate(primary_result.scores):
                if score >= self.thresholds["modal_primary"]:
                    surviving_texts.append(texts[idx])
                    surviving_indices.append(idx)
                else:
                    report["dropped_by"].append({"text_index": idx, "reason": f"{primary_result.method_name}_low_grounding", "score": score})
                    
            if surviving_texts and self.secondary_cross_modal:
                secondary_result = self.secondary_cross_modal.evaluate(image_path, surviving_texts)
                final_survivors = []
                for f_idx, score in enumerate(secondary_result.scores):
                    original_idx = surviving_indices[f_idx]
                    if score >= self.thresholds["modal_secondary"]:
                        final_survivors.append(surviving_texts[f_idx])
                    else:
                        report["dropped_by"].append({"text_index": original_idx, "reason": f"{secondary_result.method_name}_low_grounding", "score": score})
                return final_survivors
                
            return surviving_texts

        else:
            raise ValueError("cross_modal_style must be 'fallback' or 'sequential'")
        
    def _run_or_modal_first(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Runs Cross-Modal check. If a text fails, it gets a second chance with Cross-Output."""
        if not self.cross_modal or not self.cross_output:
            raise ValueError("OR aggregation requires both primary methods.")

        # 1. Cross-Modal Primary Check
        modal_survivors = self._run_cross_modal(image_path, texts, report)
        
        # 2. Identify which texts failed the cross-modal check
        failed_texts = [t for t in texts if t not in modal_survivors]
        
        # 3. Give failures a second chance with Cross-Output
        if failed_texts:
            output_result = self.cross_output.evaluate(failed_texts)
            output_survivors = self._filter_by_output(failed_texts, output_result, report)
        else:
            output_survivors = []
            
        # Combine and preserve original order
        final_survivors = set(modal_survivors + output_survivors)
        return [t for t in texts if t in final_survivors]

    def _run_or_output_first(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Runs Cross-Output check. If a text fails, it gets a second chance with Cross-Modal."""
        if not self.cross_modal or not self.cross_output:
            raise ValueError("OR aggregation requires both primary methods.")

        # 1. Cross-Output Primary Check
        output_result = self.cross_output.evaluate(texts)
        output_survivors = self._filter_by_output(texts, output_result, report)
        
        # 2. Identify which texts failed the cross-output check
        failed_texts = [t for t in texts if t not in output_survivors]

        # 3. Give failures a second chance with Cross-Modal
        if failed_texts:
            modal_survivors = self._run_cross_modal(image_path, failed_texts, report)
        else:
            modal_survivors = []
            
        # Combine and preserve original order
        final_survivors = set(output_survivors + modal_survivors)
        return [t for t in texts if t in final_survivors]
    
    def _run_weighted(self, image_path: str, texts: List[str], report: Dict) -> List[str]:
        """Combines scores from both primary methods to make a unified decision."""
        if not self.cross_modal or not self.cross_output:
            raise ValueError("Weighted aggregation requires both primary methods to be initialized.")
            
        modal_result = self.cross_modal.evaluate(image_path, texts)
        output_result = self.cross_output.evaluate(texts)
        
        surviving = []
        
        # Calculate the dynamic threshold scaling to the active models
        dynamic_thresh = (self.weights["modal"] * self.thresholds["modal_primary"]) + (self.weights["output"] * self.thresholds["output"])
        
        for idx, text in enumerate(texts):
            modal_score = modal_result.scores[idx]
            output_score = output_result.scores[idx]
            
            final_score = (self.weights["modal"] * modal_score) + (self.weights["output"] * output_score)
            
            if final_score >= dynamic_thresh:
                surviving.append(text)
            else:
                report["dropped_by"].append({"text_index": idx, "reason": f"weighted_score_too_low", "score": final_score})
                
        return surviving

    def _filter_by_output(self, texts: List[str], result, report: Dict) -> List[str]:
        surviving = []
        for idx, score in enumerate(result.scores):
            if score >= self.thresholds["output"]:
                surviving.append(texts[idx])
            else:
                report["dropped_by"].append({"text_index": idx, "reason": f"{result.method_name}_individual_outlier", "score": score})
        return surviving