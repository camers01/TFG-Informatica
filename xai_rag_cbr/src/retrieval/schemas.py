from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class QueryContext:
    """
    The input structured type for the Retrieval Module. 
    Represents a new user query to be matched against the Case Base.
    """
    img_path: str
    domain: str
    ai_task: str
    ai_problem_type: str
    class_value: str  # 'class' is a Python reserved keyword, so we use class_value
    library: str
    input_format: str
    xai_graph_type: str
    ai_model: str
    explainer: str
    attributes: str
    scope: str
    portability: str
    concurrency: str
    
    # field(init=False) because we don't pass this when creating the object; 
    # it is calculated automatically inside __post_init__.
    analytical_family: str = field(init=False) 

    def __post_init__(self):
        """
        Automatically executed immediately after the object is created.
        This isolates the logic for calculating the analytical family.
        """
        # Create the tuple matching the logic defined in 01_gen_analytical_family.py
        mapping_key = (self.xai_graph_type, self.library, self.scope)
        
        # Dictionary containing the tuples mapped to their analytical families.
        family_map = {
            # Family 1: Local_Attribution
            ("Waterfall", "SHAP", "Local"): "Local_Attribution",
            ("Force", "SHAP", "Local"): "Local_Attribution",
            ("Bar", "SHAP", "Local"): "Local_Attribution",
            ("Bar", "LIME", "Local"): "Local_Attribution",
            ("Dashboard", "LIME", "Local"): "Local_Attribution",

            # Family 2: Global_Summary
            ("Beeswarm", "SHAP", "Global"): "Global_Summary",
            ("Violin", "SHAP", "Global"): "Global_Summary",
            ("Bar", "SHAP", "Global"): "Global_Summary",

            # Family 3: Dependence_Curve
            ("Scatter", "SHAP", "Global"): "Dependence_Curve",
            ("1D", "ALE", "Global"): "Dependence_Curve",
            ("2D", "ALE", "Global"): "Dependence_Curve",

            # Family 4: Cohort_Pattern
            ("Bar", "SHAP", "Cohort"): "Cohort_Pattern",
            ("Decision", "SHAP", "Cohort"): "Cohort_Pattern",

            # Family 5: Dense_Tracking
            ("Heatmap", "SHAP", "Global"): "Dense_Tracking",
        }
        
        # Automatically assign the family, defaulting to Unknown to catch mapping errors
        self.analytical_family = family_map.get(mapping_key, "Unknown_Family")


@dataclass
class RetrievedCase:
    """
    The output structured type for the Retrieval Orchestrator.
    Contains the final scores and the complete original context.
    """
    case_id: str
    final_score: float
    tabular_score: float
    visual_score: float
    
    # This dictionary will hold the entire row from the CSV.
    metadata: Dict[str, Any]