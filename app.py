
from pathlib import Path
import tempfile

import streamlit as st
import torch
from PIL import Image

from malaria_utils import MalariaCNN, get_device, predict_image

# Project settings
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "malaria_cnn_best.pth"

st.set_page_config(
    page_title="Malaria Cell Classification",
    page_icon="🔬",
    layout="centered"
)


@st.cache_resource
def load_model():
    device = get_device()

    model = MalariaCNN().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


st.title("🔬 Malaria Cell Classification")
st.write(
    "Upload a blood-cell image to see the model's prediction."
)

st.info(
    "Educational machine-learning demonstration only. "
    "This application is not a medical diagnostic tool."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model, device = load_model()
    st.success(f"Model loaded successfully! Device: {device}")
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a cell image",
    type=["jpg", "jpeg", "png"],
    key="cell_image_upload"
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded blood-cell image",
        width=300
    )

    if st.button("Classify Image", type="primary"):

        with st.spinner("Analyzing image..."):

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".png",
                    delete=False
                ) as temp_file:
                    image.save(temp_file.name)
                    temp_path = temp_file.name

                result = predict_image(
                    image_path=temp_path,
                    model=model,
                    device=device

                )

                st.subheader("Prediction Result")

                st.write(
                    f"**Predicted class:** "
                    f"{result['predicted_class']}"
                )

                st.write(
                    f"**Model probability:** "
                    f"{result['probability']:.2%}"
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

            finally:
                if "temp_path" in locals():
                    Path(temp_path).unlink(missing_ok=True)
