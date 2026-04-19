from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import Settings
from src.inference.pipeline import InferencePipeline
from src.visualization.draw import draw_predictions


def main() -> None:
    st.set_page_config(page_title="Kleidungs-Erkennung", layout="centered")
    st.title("Kleidungs-Erkennung (Schulprojekt)")

    settings = Settings()
    detector_mode = st.selectbox(
        "Detector",
        options=["mock", "yolo"],
        index=0 if settings.detector_type == "mock" else 1,
    )
    pipeline = InferencePipeline(settings=settings, detector_type=detector_mode)

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Bitte ein Bild auswählen.")
        return

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError:
        st.error("Dateiformat nicht lesbar (bitte JPG oder PNG verwenden).")
        return
    except OSError as error:
        st.error(f"Bild konnte nicht geöffnet werden: {error}")
        return
    predictions = pipeline.predict(image)
    result_image = draw_predictions(image, predictions)

    st.subheader("Ergebnis")
    st.image(result_image, caption="Vorhersage mit Bounding Boxes", use_container_width=True)
    st.write("Vorhersagen:", predictions)


if __name__ == "__main__":
    main()
