from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.prediction.pipeline import ImagePredictionPipeline
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models/model_files"

pipeline = ImagePredictionPipeline(
    hf_model_path='google/path-foundation',
    joblib_model_path= MODEL_DIR / 'optimized_mlp_model.joblib'
)


@router.post('/predict', tags=['predictor'])
async def predict_tumor(file: UploadFile = File(...)):
    """
    Endpoint to predict the class and probabilities of an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: Predicted class and probabilities.
    """
    file_bytes = BytesIO(await file.read())
    try:
        result = pipeline.predict(file_bytes)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
