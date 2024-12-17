import tensorflow as tf
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from huggingface_hub import from_pretrained_keras, login
from .model_arch import EmbeddingClassifier
from dotenv import load_dotenv
from keras.layers import TFSMLayer
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
path_foundation = os.path.join(current_dir,"path-foundation")
lung_cls_path = os.path.join(current_dir,"lung_cancer_classifier.pth")
colon_cls_path = os.path.join(current_dir,"colon_cancer_classifier.pth")
class LungPrediction:
    def __init__(self,hf_model_path = path_foundation):
        self.hf_model_path = hf_model_path
        self.infer = self._load_huggingface_model()
        self.model_lung = self.load_model(lung_cls_path)
        self.model_colon = self.load_model(colon_cls_path,lung = False)
        self.device = torch.device("cpu")

    def _load_huggingface_model(self) -> tf.keras.Model:
        """Loads the Hugging Face model."""
        print("Loading Hugging Face model...")
        return TFSMLayer(self.hf_model_path, call_endpoint="serving_default")
    
    def get_embedding(self,image_data,target_size = (224,224)):
        image_bytes = BytesIO(image_data) 

        image = Image.open(image_bytes).convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)  # Resize the image
        image_array = np.array(image)

        # Convert to a tensor and normalize
        tensor = tf.cast(tf.expand_dims(image_array, axis=0), tf.float32) / 255.0
        embedding = self.infer(tf.constant(tensor))

        embedding_vector = torch.tensor(embedding["output_0"].numpy().flatten())
        return embedding_vector

    def load_model(self,model_path,lung = True):
        input_dim = 384
        if lung == False:
            num_classes = 2
        else:
            num_classes = 3
        model = EmbeddingClassifier(input_dim = input_dim,num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device("cpu"))
        model.eval()
        return model


    def get_preds_lung(self,embedding_vector):
        with torch.no_grad():
            embedding_vector = embedding_vector.to(self.device)
            print("1")
            embedding_vector = embedding_vector.unsqueeze(0)
            print("2")
            output = self.model_lung(embedding_vector)
            print("3")
            probs = torch.softmax(output, dim = 1)
            probs_percentages = probs.squeeze().cpu().numpy() * 100
            probs_percentages = np.round(probs_percentages, 2)
            probs_percentages = dict(zip(["Lung Benign", "Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"], probs_percentages))
            # print(probs_percentages)
            # result = torch.argmax(output, dim = 1)
            print(probs_percentages)
    def get_preds_colon(self,embedding_vector):
        with torch.no_grad():
            embedding_vector = embedding_vector.to(self.device)
            print("1")
            embedding_vector = embedding_vector.unsqueeze(0)
            print("2")
            output = self.model_colon(embedding_vector)
            print("3")
            probs = torch.softmax(output, dim = 1)
            probs_percentages = probs.squeeze().cpu().numpy() * 100
            probs_percentages = np.round(probs_percentages, 2)
            probs_percentages = dict(zip(["Benign", "Colon Adenocarcinoma"], probs_percentages))
            print(probs_percentages)
            # print(probs_percentages)
            # result = torch.argmax(output, dim = 1)
            return probs_percentages



