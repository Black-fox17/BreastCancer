from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.prediction.pipeline import ImagePredictionPipeline
from pathlib import Path
from PIL import Image

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models/model_files"

pipeline = ImagePredictionPipeline(
    hf_model_path='google/path-foundation',
    joblib_model_path= MODEL_DIR / 'optimized_mlp_model.joblib'
)


@router.post('/predict', tags=['predictor'])
def predict_tumor(file: UploadFile = File(...)):
    """
    Endpoint to predict the class and probabilities of an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: Predicted class and probabilities.
    """
    try:
        image = Image.open(file.file)
        tumor_class, class_probabilities, _ = pipeline.predict(image)
        result = {
            'predicted_class': tumor_class,
            'probabilities': class_probabilities,
        }
        return JSONResponse(content = result)
    except Exception as e:
        return HTTPException(status_code=400, detail = f'Error processing Image: {e}')
