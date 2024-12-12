import os
from huggingface_hub import from_pretrained_keras
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global variables for models
hf_model = None
joblib_model = None


def load_models(hf_model_path: str, joblib_model_path: str):
    """
    Load the Hugging Face and joblib models.

    Args:
        hf_model_path (str): Path to the Hugging Face model.
        joblib_model_path (str): Path to the joblib model.
    """
    try:
        global hf_model, joblib_model
        hf_model = from_pretrained_keras(hf_model_path, compile=False, return_full_path=True)
        joblib_model = joblib.load(joblib_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def get_hf_model():
    """
    Retrieve the loaded Hugging Face model.
    """
    return hf_model


def get_joblib_model():
    """
    Retrieve the loaded joblib model.
    """
    return joblib_model
