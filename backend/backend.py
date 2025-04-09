from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import io
import time
import os

from style_transfer_fast import load_model as load_fast_model, apply_style_transfer as apply_fast_transfer
from style_transfer_custom import load_vgg19_model, apply_style_transfer as apply_custom_transfer
from utils import preprocess_image, deprocess_image

app = FastAPI()

MODEL_PATHS = {
    "candy": "models/candy-9.onnx",
    "mosaic": "models/mosaic-9.onnx",
    "udnie": "models/udnie-9.onnx",
    "rain_princess": "models/rain-princess-9.onnx",
    "pointilism": "models/pointilism-9.onnx",
    "custom": "models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
}


@app.post("/transfer")
async def transfer_style(
    content_file: UploadFile = File(...),
    style_name: str = Form(...),
    style_file: UploadFile = File(None),
    alpha: float = Form(1.0)
):
    print(f"[INFO] Requisição recebida - estilo: '{style_name}', alpha: {alpha}")

    # Validação de estilo
    if style_name not in MODEL_PATHS:
        return JSONResponse(status_code=400, content={"error": f"Estilo '{style_name}' não é suportado."})

    # Validação de caminho do modelo
    model_path = MODEL_PATHS[style_name]
    if not os.path.exists(model_path):
        return JSONResponse(status_code=500, content={"error": f"Modelo não encontrado: {model_path}"})

    try:
        start = time.time()
        content_image, original_size = preprocess_image(content_file.file)

        if style_name == "custom":
            if not style_file:
                return JSONResponse(status_code=400, content={"error": "Imagem de estilo é obrigatória para a opção 'custom'."})
            style_image, _ = preprocess_image(style_file.file)
            model = load_vgg19_model(model_path)
            stylized = apply_custom_transfer(model, content_image, style_image, alpha)
        else:
            model = load_fast_model(model_path)
            stylized = apply_fast_transfer(model, content_image, original_size)

        result_image = deprocess_image(stylized)
        result_image = result_image.resize(original_size)

        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)

        print(f"[INFO] Transferência finalizada em {time.time() - start:.2f}s")
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        print(f"[ERRO] Falha ao aplicar o estilo: {e}")
        return JSONResponse(status_code=500, content={"error": "Erro interno ao processar a imagem."})

@app.get("/ping")
def ping():
    return {"message": "pong"}
