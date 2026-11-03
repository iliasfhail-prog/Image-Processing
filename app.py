import io

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="Sobel Edge Lab", page_icon="🖼️", layout="wide")
st.title("Sobel Edge Lab")
st.write(
    "Upload a JPEG or PNG to see the original, grayscale, horizontal edges, "
    "vertical edges, and combined edges."
)

MAX_DIMENSION = 1200
MAX_FILE_BYTES = 10 * 1024 * 1024


def sobel_edges(gray: np.ndarray):
    # Kx measures left-to-right changes (vertical edges); Ky measures
    # top-to-bottom changes (horizontal edges).
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    padded = np.pad(gray.astype(np.float32), 1, mode="edge")
    windows = sliding_window_view(padded, (3, 3))
    gx = np.einsum("ijkl,kl->ij", windows, kx)
    gy = np.einsum("ijkl,kl->ij", windows, ky)

    def normalize(values):
        maximum = values.max()
        if maximum == 0:
            return np.zeros(values.shape, dtype=np.uint8)
        return np.clip(values * (255.0 / maximum), 0, 255).astype(np.uint8)

    return (
        normalize(np.abs(gy)),
        normalize(np.abs(gx)),
        normalize(np.hypot(gx, gy)),
    )


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    if uploaded.size > MAX_FILE_BYTES:
        st.error("Please choose an image smaller than 10 MB.")
    else:
        try:
            original_bytes = uploaded.getvalue()
            with Image.open(io.BytesIO(original_bytes)) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")

            # Cap processing resolution so large phone photos remain responsive.
            original.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
            grayscale = ImageOps.grayscale(original)
            horizontal, vertical, combined = sobel_edges(np.asarray(grayscale))

            st.subheader("Results")
            columns = st.columns(4)
            results = [
                ("Original", original_bytes, None),
                ("Grayscale", grayscale, "grayscale.png"),
                ("Horizontal edges", Image.fromarray(horizontal), "horizontal_edges.png"),
                ("Vertical edges", Image.fromarray(vertical), "vertical_edges.png"),
            ]

            for column, (label, image, filename) in zip(columns, results):
                with column:
                    st.image(image, caption=label, width="stretch")
                    if filename:
                        st.download_button(
                            f"Download {label.lower()}",
                            data=png_bytes(image),
                            file_name=filename,
                            mime="image/png",
                            key=filename,
                        )

            combined_image = Image.fromarray(combined)
            st.image(combined_image, caption="Combined edges", width="stretch")
            st.download_button(
                "Download combined edges",
                data=png_bytes(combined_image),
                file_name="combined_edges.png",
                mime="image/png",
                key="combined_edges.png",
            )
            st.caption(
                "Images larger than 1200 pixels on either side are resized for faster processing. "
                "Horizontal edges come from top-to-bottom brightness changes; "
                "vertical edges come from left-to-right changes."
            )
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Could not process this image: {error}")
