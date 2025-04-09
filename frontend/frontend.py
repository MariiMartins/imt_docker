import streamlit as st
import requests
import io
import time
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Transferência de Estilo", layout="centered")
st.title("🎨 Transferência de Estilo Neural")

# Estilos disponíveis
available_styles = ["candy", "mosaic", "udnie", "rain_princess", "pointilism", "custom"]
style_name = st.selectbox("Escolha um estilo:", available_styles)

# Upload da imagem de conteúdo
content_file = st.file_uploader("Envie a imagem de conteúdo", type=["jpg", "png", "jpeg"])

# Upload da imagem de estilo, se "custom" for selecionado
style_file = None
alpha = 1.0
if style_name == "custom":
    style_file = st.file_uploader("Envie a imagem de estilo", type=["jpg", "png", "jpeg"])
    alpha = st.slider("Intensidade do estilo (alpha)", 0.0, 1.0, 0.5, step=0.01)

# Mostra imagens de entrada
if content_file:
    content_image = Image.open(content_file)
    st.subheader("📷 Imagem de conteúdo:")
    st.image(content_image, use_container_width=True)

if style_name == "custom" and style_file:
    style_image = Image.open(style_file)
    st.subheader("🎨 Imagem de estilo:")
    st.image(style_image, use_container_width=True)

# Botão para aplicar o estilo
if st.button("Aplicar Estilo"):
    if not content_file:
        st.warning("Por favor, envie a imagem de conteúdo.")
    elif style_name == "custom" and not style_file:
        st.warning("Por favor, envie também a imagem de estilo para a opção 'custom'.")
    else:
        st.info("🖌️ Processando a imagem... Aguarde...")
        start_time = time.time()

        files = {"content_file": content_file.getvalue()}
        if style_name == "custom":
            files["style_file"] = style_file.getvalue()

        data = {"style_name": style_name, "alpha": str(alpha)}

        response = requests.post("http://backend:8000/transfer", files=files, data=data)

        elapsed = int(time.time() - start_time)

        if response.status_code == 200:
            result_image = Image.open(io.BytesIO(response.content))
            st.subheader("🖼️ Imagem transformada:")
            st.image(result_image, use_container_width=True)
            st.success(f"✅ Estilo aplicado em {elapsed // 60}m {elapsed % 60}s!")
        else:
            st.error("Erro ao processar a imagem. Verifique o backend.")
