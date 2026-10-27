import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from keras.models import load_model

st.set_page_config(page_title="Age group prediction", page_icon="🙂", layout="centered")

class_names = ["0-12", "13-19", "20-29", "30-44", "45-59", "60+"]

# a little styling on top of the theme in .streamlit/config.toml
st.markdown("""
<style>
h1 { color: #5b3fa0; }
.result-card {
    background: #efe9fb;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 12px 0;
}
.result-card .label { color: #6b5b95; font-size: 0.9rem; }
.result-card .value { color: #5b3fa0; font-size: 2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_model(os.path.join(here, "age_model.keras"))


model = get_model()

st.title("Age group prediction")
st.write("Upload a close-up face image or take a photo, and the model will guess the age group.")

tab_upload, tab_camera = st.tabs(["Upload", "Take a photo"])

with tab_upload:
    uploaded = st.file_uploader("Face image", type=["jpg", "jpeg", "png"])

with tab_camera:
    photo = st.camera_input("Take a photo")

# use whichever one was provided (the photo wins if both are there)
source = photo if photo is not None else uploaded

if source is not None:
    img = Image.open(source).convert("RGB")
    st.image(img, caption="Your image", width=250)

    # same preprocessing as training: 64x64, scale to 0-1, add the batch axis
    x = np.array(img.resize((64, 64))).astype("float32") / 255
    x = x.reshape(1, 64, 64, 3)

    probs = model.predict(x, verbose=0)[0]
    best = int(probs.argmax())

    st.markdown(f"""
    <div class="result-card">
        <div class="label">Predicted age group</div>
        <div class="value">{class_names[best]}</div>
        <div class="label">Confidence: {probs[best]:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    chart_df = pd.DataFrame({"probability": probs}, index=class_names)
    st.bar_chart(chart_df, color="#9b7fd4")

    st.caption("This is a guess. The model is right about half of the time, and it works best on close-up faces.")