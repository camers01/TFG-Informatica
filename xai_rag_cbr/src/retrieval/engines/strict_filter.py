import pandas as pd
from src.retrieval.schemas import QueryContext
from .base import BaseEngine

class StrictFilterEngine(BaseEngine):
    """
    Applies hard boolean masks to the database based on 
    the selected columns of the user's query.
    """
    def execute(self, df: pd.DataFrame, query: QueryContext) -> pd.DataFrame:

        self.log(f"Received {len(df)} total cases in database.")
        
        mask = (
            (df['input_format'] == query.input_format) &
            (df['ai_problem_type'] == query.ai_problem_type) &
            (df['scope'] == query.scope) &
            (df['concurrency'] == query.concurrency) &
            (df['portability'] == query.portability)
        )

        filtered_df = df[mask].copy()
        
        self.log(f"Cases remaining after Strict Filtering: {len(filtered_df)}")

        return filtered_df.copy()