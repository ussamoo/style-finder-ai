# models/image_processor.py
"""
Module for image processing, encoding, and similarity matching.
This module handles the conversion of images to vectors and finding
similar images in the dataset.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests
import base64
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImageProcessor:
    """
    Handles image processing, encoding, and similarity comparisons.
    """
    
    def __init__(self, image_size=(224, 224), 
                 norm_mean=[0.485, 0.456, 0.406], 
                 norm_std=[0.229, 0.224, 0.225]):
        """
        Initialize the image processor with a pre-trained ResNet50 model.
        
        Args:
            image_size (tuple): Target size for input images
            norm_mean (list): Normalization mean values for RGB channels
            norm_std (list): Normalization standard deviation values for RGB channels
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    
    def encode_image(self, image_input, is_url=True):
        """
        Encode an image and extract its feature vector.
        
        Args:
            image_input: URL or local path to the image
            is_url: Whether the input is a URL (True) or a local file path (False)
            
        Returns:
            dict: Contains 'base64' string and 'vector' (feature embedding)
        """
        try:

            if is_url:
                # Fetch the image from URL
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Load the image from a local file
                image = Image.open(image_input).convert("RGB")
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
            
            # TODO: Preprocess the image for ResNet50
            # Hint: Use the preprocess pipeline and add a batch dimension with unsqueeze(0)
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(input_tensor)

            feature_vector = features.cpu().numpy().flatten()
            
            return {"base64": base64_string, "vector": feature_vector}
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}

    def find_closest_match(self, user_vector, dataset):
        """
        Find the closest match in the dataset based on cosine similarity.
        
        Args:
            user_vector: Feature vector of the user-uploaded image
            dataset: DataFrame containing precomputed feature vectors
            
        Returns:
            tuple: (Closest matching row, similarity score)
        """
        try:

            dataset_vectors = np.vstack(dataset['Embedding'].dropna().values)
            
            similarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)
            
            closest_index = np.argmax(similarities)
            similarity_score = similarities[0][closest_index]
            
            # Retrieve the closest matching row
            closest_row = dataset.iloc[closest_index]
            return closest_row, similarity_score
        except Exception as e:
            print(f"Error finding closest match: {e}")
            return None, None