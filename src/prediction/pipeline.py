import os
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import from_pretrained_keras, login
from dotenv import load_dotenv
import joblib
from pathlib import Path

# Load environment variables
load_dotenv()
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
        return from_pretrained_keras(self.hf_model_path, compile=False, return_full_path=True)

    def _load_joblib_model(self):
        """Loads the classification model saved with joblib."""
        print("Loading joblib model...")
        return joblib.load(self.joblib_model_path)

    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        """
        Preprocesses an image for embedding generation.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image as a 4D tensor.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            return np.expand_dims(image_array, axis=0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")

    def generate_embedding(self, image_path: str) -> np.ndarray:
        """
        Generates an embedding for a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Generated embedding.
        """
        preprocessed_image = self.preprocess_image(image_path)
        try:
            result = self.hf_model.signatures['serving_default'](
                tf.constant(preprocessed_image, dtype=tf.float32)
            )
            return result['output_0'].numpy().flatten()  # Flatten to 1D
        except Exception as e:
            raise ValueError(f"Error generating embedding for image {image_path}: {e}")

    def predict(self, image_path: str):
        """
        Predicts the class and probabilities for a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Predicted class label, class-probability mapping, raw probabilities.
        """
        # Generate embedding
        embedding = self.generate_embedding(image_path)
        embedding_reshaped = embedding.reshape(1, -1)  # Reshape for the model

        # Perform prediction
        predicted_class = self.joblib_model.predict(embedding_reshaped)[0]
        probabilities = self.joblib_model.predict_proba(embedding_reshaped)[0]

        # Map class labels to probabilities
        class_probabilities = {
            label: prob for label, prob in zip(self.joblib_model.classes_, probabilities)
        }

        return predicted_class, class_probabilities, probabilities

pipeline = ImagePredictionPipeline(
    hf_model_path='google/path-foundation',
    joblib_model_path= MODEL_DIR / 'optimized_mlp_model.joblib'
)

# image_path = '/home/halleluyah/Documents/Programming-Projects/organisation-projects/artificial-intelligence/src/test_images/temp_sampled/adenosis/SOB_B_A-14-22549AB-40-015.png'
# predicted_class, class_probabilities, raw_probabilities = pipeline.predict(image_path)

# print(f"Predicted Class: {predicted_class}")
# print("Class Probabilities:")
# for cls, prob in class_probabilities.items():
#   print(f"  Class: {cls}, Probability: {prob:.4f}")
