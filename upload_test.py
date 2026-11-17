
import streamlit as st
from PIL import Image
import io

st.title("Malaria Image Upload Test")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    key="test_upload"
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.success("Image uploaded and opened successfully!")
        st.write("File size:", len(image_bytes), "bytes")
        st.write("Image dimensions:", image.size)
        st.image(image)
    except Exception as e:
        st.error(f"Image error: {e}")

