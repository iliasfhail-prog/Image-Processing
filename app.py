import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="Sobel Edge Lab", page_icon="🖼️", layout="wide")

st.title("Sobel Edge Lab")
st.write("Upload a JPEG or PNG to generate grayscale, horizontal-edge, vertical-edge, and combined-edge images.")

def sobel_edges(gray: np.ndarray):
    # Sobel kernels. Kx responds to vertical edges; Ky responds to horizontal edges.
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    gx = np.einsum("ijkl,kl->ij", windows, kx)
    gy = np.einsum("ijkl,kl->ij", windows, ky)

    # Display each directional response as a positive magnitude image.
    vertical_edges = np.abs(gx)
    horizontal_edges = np.abs(gy)
    combined = np.hypot(gx, gy)

    def normalize(x):
        maximum = x.max()
        if maximum == 0:
            return np.zeros_like(x, dtype=np.uint8)
        return np.clip((x / maximum) * 255, 0, 255).astype(np.uint8)

    return normalize(horizontal_edges), normalize(vertical_edges), normalize(combined)

def png_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        original = Image.open(uploaded).convert("RGB")
        grayscale = ImageOps.grayscale(original)
        gray_array = np.asarray(grayscale)
        horizontal, vertical, combined = sobel_edges(gray_array)

        horizontal_img = Image.fromarray(horizontal)
        vertical_img = Image.fromarray(vertical)
        combined_img = Image.fromarray(combined)

        st.subheader("Results")
        columns = st.columns(4)
        results = [
            ("Original", original, None),
            ("Grayscale", grayscale, "grayscale.png"),
            ("Horizontal edges", horizontal_img, "horizontal_edges.png"),
            ("Vertical edges", vertical_img, "vertical_edges.png"),
        ]

        for column, (label, image, filename) in zip(columns, results):
            with column:
                st.image(image, caption=label, use_container_width=True)
                if filename:
                    st.download_button(
                        f"Download {label.lower()}",
                        png_bytes(image),
                        filename,
                        "image/png",
                        key=filename,
                    )

        st.image(combined_img, caption="Combined edges", use_container_width=True)
        st.download_button(
            "Download combined edges",
            png_bytes(combined_img),
            "combined_edges.png",
            "image/png",
            key="combined_edges.png",
        )

        st.info(
            "Naming note: the horizontal-edge image highlights horizontal lines. "
            "The vertical Sobel kernel produces that result because it measures "
            "intensity changes across horizontal edges."
        )
    except Exception as error:
        st.error(f"Could not process this image: {error}")
