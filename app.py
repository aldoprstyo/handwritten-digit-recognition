import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps, ImageFilter

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢", layout="centered")

class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_model():
    model = DigitModel()
    state = torch.load("digit_model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def prepare_image(uploaded_image):
    image = Image.open(uploaded_image).convert("L")
    # Standarkan menjadi tulisan terang di latar gelap seperti dataset Digits.
    arr = np.asarray(image)
    if arr.mean() > 127:
        image = ImageOps.invert(image)
    image = image.point(lambda p: 0 if p < 35 else p)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image.thumbnail((6, 6))
    canvas = Image.new("L", (8, 8), 0)
    x = (8 - image.width) // 2
    y = (8 - image.height) // 2
    canvas.paste(image, (x, y))
    arr = np.asarray(canvas, dtype=np.float32)
    # Dataset sklearn digits memakai rentang 0-16.
    arr = np.clip(arr / 255.0 * 16.0, 0, 16) / 16.0
    return canvas, torch.tensor(arr.reshape(1, 64), dtype=torch.float32)

st.title("🔢 Handwritten Digit Recognition")
st.write("Aplikasi sederhana untuk mengenali angka tulisan tangan 0 sampai 9 menggunakan model PyTorch.")
st.info("Unggah gambar satu angka. Gunakan tulisan gelap di atas latar putih agar hasil lebih baik.")

uploaded = st.file_uploader("Pilih gambar angka (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    try:
        preview, tensor = prepare_image(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Gambar input")
            st.image(uploaded, use_container_width=True)
        with col2:
            st.subheader("Setelah diproses (8×8)")
            st.image(preview.resize((160,160)), caption="Pratinjau input model", use_container_width=True)
        model = load_model()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            confidence, predicted = torch.max(probabilities, dim=0)
        st.success(f"Prediksi angka: **{predicted.item()}**")
        st.metric("Tingkat keyakinan model", f"{confidence.item()*100:.1f}%")
        st.caption("Catatan: tingkat keyakinan bukan jaminan bahwa prediksi selalu benar.")
    except Exception as e:
        st.error(f"Gambar tidak dapat diproses: {e}")
else:
    st.markdown("### Cara menggunakan")
    st.markdown("1. Tulis satu angka besar pada kertas putih.\\n2. Foto atau pindai angka tersebut.\\n3. Unggah gambar melalui tombol di atas.\\n4. Lihat hasil prediksi.")
st.divider()
st.caption("Proyek Tugas Besar Machine Learning | Dataset: sklearn load_digits")
