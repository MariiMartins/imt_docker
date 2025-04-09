import onnxruntime as ort
import numpy as np
from utils import preprocess_image, deprocess_image

print("Módulo style_transfer_fast (ONNX) carregado...")

def load_model(model_path):
    print(f"Carregando modelo ONNX: {model_path}")
    return ort.InferenceSession(model_path)

def apply_style_transfer(model, content_image, original_size):
    print("Aplicando estilo...")

    # content_image já está shape (1, 224, 224, 3)
    image = np.transpose(content_image, (0, 3, 1, 2)).astype(np.float32)

    outputs = model.run(None, {"input1": image})
    stylized_image = np.transpose(outputs[0], (0, 2, 3, 1))

    pil_image = deprocess_image(stylized_image)
    resized_image = pil_image.resize(original_size)

    return resized_image


