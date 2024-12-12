from PIL import Image
import numpy as np


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses an image for embedding generation.

    Args:
        image (Image.Image): The PIL image.

    Returns:
        np.ndarray: Preprocessed image as a 4D tensor.
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # Add batch dimension
