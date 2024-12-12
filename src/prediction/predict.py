from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from src.prediction.preprocess import preprocess_image
from src.prediction.predict import predict

router = APIRouter()


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to predict the class and probabilities of an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: Predicted class and probabilities.
    """
    try:
        image = Image.open(file.file)
        preprocessed_image = preprocess_image(image)
        predicted_class, class_probabilities, _ = predict(preprocessed_image)

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "class_probabilities": class_probabilities
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
