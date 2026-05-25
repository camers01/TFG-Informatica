"""
Configuration parameters and weights for the Retrieval Module.
Centralizing these values allows for easy tuning without touching engine logic.
"""

# 1. ORCHESTRATION WEIGHTS

FUSION_WEIGHT_Z = 0.65
TOP_N_VISUAL_CANDIDATES = 20
TOP_K_FINAL_RESULTS = 3


# 2. TABULAR SCORING WEIGHTS

TABULAR_WEIGHTS = {
    # HIGH
    "analytical_family": 0.40, 
    # MEDIUM
    "domain": 0.15,            
    "ai_task": 0.15,           
    "ai_model": 0.15,          
    # LOW
    "attributes": 0.05,        
    "class_value": 0.05,       
    "explainer": 0.05          
}


# 3. AI MODEL HIERARCHY

MODEL_SUB_FAMILIES = {
    # Tree Ensembles
    "RandomForestRegressor": "Tree_Ensemble",
    "RandomForestClassifier": "Tree_Ensemble",
    "GradientBoostingRegressor": "Tree_Ensemble",
    "GradientBoostingClassifier": "Tree_Ensemble",
    "ExtraTreesClassifier": "Tree_Ensemble",
    # Single Trees
    "DecisionTreeRegressor": "Single_Tree",
    "DecisionTreeClassifier": "Single_Tree",
    # Linear Models
    "LinearRegression": "Linear",
    "LogisticRegression": "Linear",
    "Ridge": "Linear",
    "Lasso": "Linear",
    # Distance / Geometric
    "KNeighborsRegressor": "Distance",
    "KNeighborsClassifier": "Distance",
    "SVR": "Distance",
    "SVC": "Distance",
    # Neural Networks / Deep Learning
    "MLPRegressor": "Neural_Network",
    "MLPClassifier": "Neural_Network",
    "Keras_DNN": "Neural_Network",
    "TorchRegressor": "Neural_Network",
    "TorchClassifier": "Neural_Network",
    "TorchMulticlass": "Neural_Network"
}

MODEL_BROAD_CATEGORIES = {
    "Tree_Ensemble": "Tree_Based",
    "Single_Tree": "Tree_Based",
    "Linear": "Mathematical",
    "Distance": "Mathematical",
    "Neural_Network": "Deep_Learning"
}

def get_model_score(query_val: str, db_val: str) -> float:
    """
    Evaluates AI Model similarity.
    Level 0 (Exact Match): 1.0
    Level 1 (Same Sub-Family): 0.7
    Level 2 (Same Broad Category): 0.3
    Mismatch: 0.0
    """
    if query_val == db_val:
        return 1.0
        
    sub_q = MODEL_SUB_FAMILIES.get(query_val)
    sub_db = MODEL_SUB_FAMILIES.get(db_val)
    
    if sub_q and sub_db and (sub_q == sub_db):
        return 0.7
        
    broad_q = MODEL_BROAD_CATEGORIES.get(sub_q)
    broad_db = MODEL_BROAD_CATEGORIES.get(sub_db)
    
    if broad_q and broad_db and (broad_q == broad_db):
        return 0.3
        
    return 0.0


# 4. EXPLAINER HIERARCHY

EXPLAINER_NATURE = {
    # Model-Specific
    "TreeExplainer": "Path_Dependent",
    "DeepExplainer": "Gradient_Based",
    "GradientExplainer": "Gradient_Based",
    "LinearExplainer": "Weight_Based",
    # Model-Agnostic Perturbation
    "KernelExplainer": "Perturbation",
    "PermutationExplainer": "Perturbation",
    "LimeTabularExplainer": "Perturbation",
    # Model-Agnostic Distribution
    "PyALE_Function": "Distribution_Grid"
}

def get_explainer_score(query_val: str, db_val: str) -> float:
    """
    Evaluates Explainer similarity.
    Level 0 (Exact Match): 1.0
    Level 1 (Same Nature): 0.6
    Mismatch: 0.0
    """
    if query_val == db_val:
        return 1.0
        
    nature_q = EXPLAINER_NATURE.get(query_val)
    nature_db = EXPLAINER_NATURE.get(db_val)
    
    if nature_q and nature_db and (nature_q == nature_db):
        return 0.6
        
    return 0.0