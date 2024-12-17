from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from io import BytesIO
import warnings
import sys
sys.path.append('..')
from prediction.pipeline import ImagePredictionPipeline
from prediction.llm import get_response
from fastapi.middleware.cors import CORSMiddleware
warnings.filterwarnings('ignore')

router = FastAPI()


origins = ['http://localhost:3000']
router.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=['*'],
    allow_headers=['*'], 
)
 
pipeline = ImagePredictionPipeline()


@router.post('/predict', tags=['predictor'])
async def predict_tumor(file: UploadFile = File(...)):
    """
    Endpoint to predict the class and probabilities of an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: Predicted class and probabilities.
    """
    file_content = await file.read()
    file_bytes = BytesIO(file_content)
    try:
        result = pipeline.predict(file_bytes)
        response = get_response(file_content,result)
        return {"success": True, "result": response}
    except Exception as e:
        return {"success": False, "error": str(e)} 
