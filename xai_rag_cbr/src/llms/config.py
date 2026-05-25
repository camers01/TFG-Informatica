import os

def is_kaggle_environment() -> bool:
    """
    Detects if the code is running inside a Kaggle environment, as Kaggle 
    injects specific environment variables (like KAGGLE_KERNEL_RUN_TYPE) 
    into its active sessions. This is important because if it isn't running
    in Kaggle, we avoid loading heavy models on local machines that may crash.
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def get_gemini_api_key() -> str:
    """
    Smartly retrieves the API key depending on the environment.
    Uses Kaggle Secrets in the cloud, and .env locally.
    """
    if is_kaggle_environment():

        # KAGGLE CLOUD ENVIRONMENT 
        from kaggle_secrets import UserSecretsClient # type: ignore
        try:
            user_secrets = UserSecretsClient()
            return user_secrets.get_secret("API_KEY_GEMINI")
        except Exception as e:
            raise ValueError(f"Failed to load Kaggle Secret. Did you attach it to the notebook? Error: {e}")
            
    else:

        # LOCAL VS CODE ENVIRONMENT
        from dotenv import load_dotenv
        
        # This automatically finds your .env file and loads the variables
        load_dotenv() 
        
        api_key = os.getenv("API_KEY_GEMINI")
        if not api_key:
            raise ValueError("API_KEY_GEMINI not found in local .env file.")
        
        return api_key