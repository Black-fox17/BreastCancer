import os
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import from_pretrained_keras, login
from dotenv import load_dotenv
import joblib
from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models/model_files"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for compatibility if needed

# Login to Hugging Face
login(token=os.getenv('HUGGINGFACE_HUB_API_KEY'))


class ImagePredictionPipeline:
    def __init__(self, hf_model_path: str, joblib_model_path: str):
        """
        Initializes the prediction pipeline.

        Args:
            hf_model_path (str): Path to the Hugging Face model on the Hub.
            joblib_model_path (str): Path to the joblib model for classification.
        """
        self.hf_model_path = hf_model_path
        self.joblib_model_path = joblib_model_path
        self.hf_model = self._load_huggingface_model()
        self.joblib_model = self._load_joblib_model()

    def _load_huggingface_model(self) -> tf.keras.Model:
        """Loads the Hugging Face model."""
        print("Loading Hugging Face model...")
        return from_pretrained_keras(self.hf_model_path, compile=False)

    def _load_joblib_model(self):
        """Loads the classification model saved with joblib."""
        print("Loading joblib model...")
        return joblib.load(self.joblib_model_path)

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """
        Preprocesses an image for embedding generation.

        Args:
            image (PIL.Image): Image object.

        Returns:
            np.ndarray: Preprocessed image as a 4D tensor.
        """
        try:
            # Convert to RGB and resize
            image = image.convert("RGB")
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            return np.expand_dims(image_array, axis=0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generates an embedding for a single image.

        Args:
            image (PIL.Image): Image object.

        Returns:
            np.ndarray: Generated embedding.
        """
        preprocessed_image = self.preprocess_image(image)
        try:
            result = self.hf_model.signatures['serving_default'](
                tf.constant(preprocessed_image, dtype=tf.float32)
            )
            # Convert the result to a Python list
            return result['output_0'].numpy().flatten().tolist()
        except Exception as e:
            raise ValueError(f"Error generating embedding for the image: {e}")

    def predict(self, file: BytesIO):
        """
        Predicts the class and probabilities for a given image.

        Args:
            file (BytesIO): File-like object containing image data.

        Returns:
            dict: Prediction results including class, probabilities, and raw outputs.
        """
        try:
            # Open the image directly from the file stream
            image = Image.open(file)
        except Exception as e:
            raise ValueError(f"Error opening image file: {e}")

        # Generate embedding
        embedding = self.generate_embedding(image)
        embedding_reshaped = np.array(embedding).reshape(1, -1)  # Reshape for the model

        # Perform prediction
        predicted_class = self.joblib_model.predict(embedding_reshaped)[0]
        probabilities = self.joblib_model.predict_proba(embedding_reshaped)[0]

        # Map class labels to probabilities
        class_probabilities = {
            label: float(prob) for label, prob in zip(self.joblib_model.classes_, probabilities)
        }

        return {
            "predicted_class": str(predicted_class),  # Return the actual predicted class
            "class_probabilities": class_probabilities,  # Map of class labels to probabilities
            "raw_probabilities": [float(prob) for prob in probabilities],  # Raw probability values
        }
