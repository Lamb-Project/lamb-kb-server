from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
import os


class CLIPEmbeddingFunction:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._name = "clip"

    def __call__(self, input):
        embeddings = []

        for item in input:
            try:
                if os.path.exists(item):
                    image = Image.open(item).convert("RGB")
                    proc_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    features = self.model.get_image_features(**proc_inputs)
                else:
                    proc_inputs = self.processor(text=item, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    features = self.model.get_text_features(**proc_inputs)

                features = features.detach().cpu().numpy().tolist()
                embeddings.append(features[0])
            except Exception as e:
                print(f"Error procesando input {item}: {e}")
                embeddings.append([0.0] * self.model.config.projection_dim)
        return embeddings

    def name(self):
        return self._name
