from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.models.load_models import load_models
from src.api.predict_route import router

app = FastAPI(
    title="Image Classification API",
    description="API for predicting image classes using pre-trained models.",
    version="1.0.0",
)

# Include routes
app.include_router(router, prefix="/api")


# @app.on_event("startup")
# async def startup_event():
#     """
#     Load models at application startup.
#     """
#     print("Loading models...")
#     hf_model_path = './models/model_files/hf_model/'
#     joblib_model_path = './models/model_files/ml_model.joblib'
#     load_models(hf_model_path, joblib_model_path)
#     print("Loaded models successfully...")


# @app.on_event("shutdown")
# async def shutdown_event():
#     """
#     Clean up resources at application shutdown.
#     """
#     from src.models.load_models import hf_model, joblib_model
#     hf_model = None
#     joblib_model = None


@app.get("/", include_in_schema=False)
async def root():
    """
    Redirect to API documentation.
    """
    return RedirectResponse(url="/docs")
