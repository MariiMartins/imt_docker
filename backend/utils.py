import numpy as np
from PIL import Image

def preprocess_image(file):
    print("Preprocessando imagem...")
    img = Image.open(file).convert("RGB")
    original_size = img.size
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)[np.newaxis, ...]
    return img, original_size

def deprocess_image(tensor):
    print("Convertendo tensor para imagem...")
    if isinstance(tensor, np.ndarray):
        tensor = np.squeeze(tensor, axis=0) if tensor.ndim == 4 else tensor
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)
