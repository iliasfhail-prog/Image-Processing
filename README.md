# Sobel Edge Lab

A small open-source image-processing app that accepts a JPEG or PNG and outputs:

- Grayscale image
- Horizontal-edge image
- Vertical-edge image
- Combined Sobel-edge image

The processing is done locally by the app using NumPy and Pillow. No AI image generation is used.

## Run locally

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Deploy publicly

1. Create a new public repository on GitHub.
2. Upload `app.py`, `requirements.txt`, `README.md`, `LICENSE`, and `.gitignore`.
3. Go to [share.streamlit.io](https://share.streamlit.io/).
4. Sign in with GitHub and choose this repository.
5. Set the main file to `app.py`.
6. Deploy and copy the public app URL.

## Create a QR code

After deployment, replace the URL in `make_qr.py` with your Streamlit app URL:

```bash
pip install qrcode[pil]
python make_qr.py
```

This creates `sobel_edge_lab_qr.png`, which can be printed or shared.

## Sobel convention

The app labels results by the orientation of the lines detected:

- Horizontal edges: produced from the vertical Sobel response (`Gy`)
- Vertical edges: produced from the horizontal Sobel response (`Gx`)
- Combined edges: `sqrt(Gx² + Gy²)`

## License

MIT License. See `LICENSE`.
