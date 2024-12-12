import numpy as np
from src.models.load_models import get_hf_model, get_joblib_model


def generate_embedding(preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Generates an embedding for a preprocessed image.

    Args:
        preprocessed_image (np.ndarray): Preprocessed image tensor.

    Returns:
        np.ndarray: Generated embedding.
    """
    hf_model = get_hf_model()
    result = hf_model.signatures['serving_default'](
        tf.constant(preprocessed_image, dtype=tf.float32)
    )
    return result['output_0'].numpy().flatten()


def predict(preprocessed_image: np.ndarray):
    """
    Predicts the class and probabilities for a given preprocessed image.

    Args:
        preprocessed_image (np.ndarray): Preprocessed image.

    Returns:
        tuple: Predicted class label, class-probability mapping, raw probabilities.
    """
    # Generate embedding
    embedding = generate_embedding(preprocessed_image).reshape(1, -1)

    # Perform prediction
    joblib_model = get_joblib_model()
    predicted_class = joblib_model.predict(embedding)[0]
    probabilities = joblib_model.predict_proba(embedding)[0]

    class_probabilities = {
        label: prob for label, prob in zip(joblib_model.classes_, probabilities)
    }

    return predicted_class, class_probabilities, probabilities
