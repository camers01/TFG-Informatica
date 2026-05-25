from typing import Dict, List, Optional

from typer import prompt

class PromptManager:
    def __init__(self, version: str = "default"):
        """
        Allows to quickly swap between different prompt engineering strategies.
        """
        self.version = version

    def build(self, case_data: Dict, retrieved_cases: Optional[List[Dict]] = None) -> str:
        """
        The only method the rest of the system needs to call.
        It routes to the correct template based on available data.
        """
        if not retrieved_cases:
            return self._build_basic_prompt(case_data)
        else:
            return self._build_rag_prompt(case_data, retrieved_cases)

    ###### BASIC ZERO-SHOT PROMPTS (For building the initial database) ######

    def _build_basic_prompt(self, row: Dict) -> str:

        if self.version == "default":

            # We use a structured prompt that guides the LLM to use the visual evidence from graph and context, using Markdown formattingto help the attention mechanisms separate instructions from contextual data
            
            return f"""
# ROLE
You are an Expert Data Scientist specializing in Explainable AI (xAI) and Machine Learning interpretability. Your task is to analyze the attached xAI visualization and extract precise, factual insights. Your analysis will serve as the natural language foundation for a second-level explanation system that justifies the model's behavior.

# CONTEXT
Here are the technical specifications for the attached graph:
* **Domain:** {row['domain']}
* **AI Task:** {row['ai_task']}
* **Problem Type:** {row['ai_problem_type']}
* **Input Data Format:** {row['input_format']}
* **Target Analyzed:** {row['class']} (Specific class, local value, or NA)
* **AI Model:** {row['ai_model']}
* **xAI Library:** {row['library']}
* **Explainer Used:** {row['explainer']}
* **Graph Type:** {row['xai_graph_type']}
* **Scope:** {row['scope']}
* **Portability:** {row['portability']}
* **Concurrency:** {row['concurrency']}
* **Available Dataset Features:** {row['attributes']} (CRITICAL NOTE: The list of features represents the universe of possible features in the dataset as a guide. DO NOT assume all of them are shown in the graph. You must visually verify which specific features are actually rendered in the image before mentioning them.)

# TASK
Analyze the attached visual graph meticulously. Do not guess or hallucinate data that is not visually evident, rely strictly on the visual evidence provided by the graph and the context above. Cross-reference the visual evidence with the provided metadata to deduce the visible features, their direction, and magnitude of impact.

# OUTPUT FORMAT
Provide your response as a SINGLE, highly professional analytical PARAGRAPH. 
You must logically justify every claim you make using direct visual evidence. Explicitly explain:
1. Exactly which features are visually present driving the decision.
2. The visual evidence (e.g., color, left/right position) that dictates their direction of impact.
3. The visual evidence (e.g., specific bar lengths, relative size differences) that justifies their magnitude of importance compared to one another. Do not simply state that one feature is more important than another; explain exactly *how* the visualization proves this.
"""
        
        # TODO: FUTURE PROMPT VERSIONS TO BE ADDED HERE
        
        else:
            raise ValueError(f"Unknown prompt version: {self.version}")

    ###### FEW-SHOT RAG PROMPTS (For the final user system) ######

    def _build_rag_prompt(self, row: Dict, retrieved_cases: List[Dict]) -> str:
        """
        Few-shot RAG prompt: appends retrieved cases to the unchanged zero-shot base.
        The zero-shot prompt is kept identical to preserve evaluation comparability.
        Retrieved examples serve ONLY as reasoning style references, not data sources.
        """

        # Zero-shot base 
        prompt = self._build_basic_prompt(row)

        # Few-shot block header
        prompt += """

---

# REFERENCE METHODOLOGY (DO NOT COPY DATA)
The following cases are retrieved from a reference database because their graph structure and analytical intent is similar to the one you must analyze.

**CRITICAL INSTRUCTIONS FOR USING THESE EXAMPLES:**
1. Observe the **analytical reasoning style**: how visual elements (colors, axis positions, bar lengths, directions) are mapped to conclusions about feature importance and direction.
2. You will notice these examples are data-dense, containing specific numerical values, target predictions, explicit feature names, and domain contexts. You MUST treat all of them as **FICTIONAL PLACEHOLDERS** — they belong to entirely different cases.
3. Do **NOT** copy, reference, or be influenced by the specific features, values, or domains from these examples.
4. Do **NOT** assume the features listed in the examples are present in your target image.
5. Your analysis must derive **exclusively** from the visual evidence in the attached image and the metadata provided above. Every number, feature name, and directional claim in your output must be verified in the attached image.

"""

        # Retrieved cases
        for i, case in enumerate(retrieved_cases, 1):
            prompt += f"### Reference Example {i}\n"
            prompt += f"- **Graph Type:** {case.get('xai_graph_type', 'Unknown')}\n"
            prompt += f"- **Analytical Family:** {case.get('analytical_family', 'Unknown')}\n"
            prompt += f"- **Explainer:** {case.get('explainer', 'Unknown')}\n"
            prompt += f"- **Problem Type:** {case.get('ai_problem_type', 'Unknown')}\n"
            prompt += f"- **Scope:** {case.get('scope', 'Unknown')}\n"
            prompt += f"**Reference Analysis Style:**\n> {case.get('solution_insights', '')}\n\n"

        # Re-anchoring to the main task (to cover recency bias)
        prompt += """
---

Now produce your analysis of the **attached image** following the exact output format specified at the beginning (single analytical paragraph). Your response must be grounded exclusively in what you can visually verify in the attached graph."""

        return prompt