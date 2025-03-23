from typing import List, Dict, Any, Union
import numpy as np
import fitz  # PyMuPDF
import ollama
from PIL import Image
import io
import base64
from transformers import CLIPProcessor, CLIPModel
import torch

class MultimodalEmbedding:
    def __init__(self):
        # Initialize CLIP model for image embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embeddings using Ollama."""
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return np.array(response["embedding"])

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate image embeddings using CLIP."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().cpu().numpy()[0]

    def combine_embeddings(self, text_embedding: np.ndarray, image_embedding: np.ndarray) -> np.ndarray:
        """Combine text and image embeddings into a unified representation."""
        # Normalize embeddings
        text_norm = text_embedding / np.linalg.norm(text_embedding)
        image_norm = image_embedding / np.linalg.norm(image_embedding)
        
        # Concatenate and normalize again
        combined = np.concatenate([text_norm, image_norm])
        return combined / np.linalg.norm(combined)

    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF with their associated text context."""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num, page in enumerate(doc):
            # Get page text
            text = page.get_text()
            
            # Extract images
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                
                # Get image location and surrounding text
                rect = page.get_image_bbox(img)
                surrounding_text = page.get_text("text", clip=rect).strip()
                
                images.append({
                    "page": page_num,
                    "image": image,
                    "context_text": surrounding_text,
                    "location": rect
                })
        
        return images

    def process_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a PDF document and generate multimodal chunks."""
        chunks = []
        
        # Extract images and their context
        images = self.extract_images_from_pdf(pdf_path)
        
        # Process each image and its context
        for img_data in images:
            # Generate embeddings
            text_embedding = self.get_text_embedding(img_data["context_text"])
            image_embedding = self.get_image_embedding(img_data["image"])
            combined_embedding = self.combine_embeddings(text_embedding, image_embedding)
            
            # Convert image to base64 for storage
            img_byte_arr = io.BytesIO()
            img_data["image"].save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode()
            
            chunks.append({
                "type": "multimodal",
                "page": img_data["page"],
                "text": img_data["context_text"],
                "image_base64": img_base64,
                "location": img_data["location"],
                "embedding": combined_embedding
            })
        
        return chunks

    def process_query(self, query: str, image: Image.Image = None) -> np.ndarray:
        """Process a query that may include both text and image."""
        if image is None:
            # Text-only query
            return self.get_text_embedding(query)
        else:
            # Multimodal query
            text_embedding = self.get_text_embedding(query)
            image_embedding = self.get_image_embedding(image)
            return self.combine_embeddings(text_embedding, image_embedding) 