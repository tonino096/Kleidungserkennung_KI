from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# Auch die Streamlit-App erweitert den Importpfad einmalig, damit das Projekt
# direkt über den app-Ordner startbar bleibt.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config.settings as settings_module
from src.detector.yolo_detector import YoloDetector
from src.inference.pipeline import InferencePipeline
from src.testing.matchtest import DeepFashion2MatchTester
from src.visualization.draw import draw_predictions

# Diese Datei baut die Weboberfläche. Streamlit führt das Skript von oben nach
# unten aus und zeichnet aus den st-Aufrufen die Seite im Browser.

# Diese Breite sorgt dafür, dass Einzelbild- und Matchtest-Ergebnisse in der
# Oberfläche konsistent groß angezeigt werden.
_STREAMLIT_IMAGE_WIDTH = 1280
# Die drei Modell-Labels werden zentral gehalten, damit Dropdowns, Fallbacks
# und Pfadzuordnungen im ganzen Streamlit-Code dieselben Namen verwenden.
_MODEL_LABEL_V8 = "YOLO v8n 15 Epochen"
_MODEL_LABEL_V11 = "YOLO v11n 15 Epochen"
_MODEL_LABEL_BEST = "YOLO best"
Settings = settings_module.Settings


def resolve_project_path(path_value: str | Path) -> Path:
    # Die Oberfläche arbeitet viel mit Nutzereingaben. Relative Pfade werden
    # hier einheitlich in Projektpfade umgerechnet.
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_available_model_names_from_settings(settings: object) -> list[str]:
    # Die Streamlit-App verwendet bewusst eine eigene, stabile Modellliste,
    # damit alte Settings-Objekte nicht mehr auf veraltete Modellnamen oder
    # Gewichtspfade wie "best.pt" zurückfallen können.
    return list(build_streamlit_model_registry().keys())


def get_weights_path_from_settings(settings: object, model_name: str | None = None) -> str:
    # Auch die Pfadauflösung wird lokal stabil gehalten, damit die Auswahl
    # immer dieselben Dateien meint, selbst wenn Streamlit noch alte Objekte
    # oder Session-Werte im Speicher hat.
    registry = build_streamlit_model_registry()
    # Ohne explizite Auswahl wird zuerst das aktive Modell aus den Settings
    # genutzt und nur im Notfall auf das v8n-Modell als Standard gewechselt.
    chosen_model = model_name or getattr(settings, "active_yolo_model", _MODEL_LABEL_V8)
    return str(registry.get(chosen_model, registry[_MODEL_LABEL_V8]))


def build_streamlit_model_registry() -> dict[str, str]:
    # Diese Registry ist absichtlich doppelt zur Settings-Datei hinterlegt.
    # So bleibt die UI robust, auch wenn noch eine ältere Settings-Klasse
    # ohne aktuelle Modellnamen oder Pfade im laufenden Prozess steckt.
    best_default = resolve_project_path("models/best.pt")
    if not best_default.exists():
        # "YOLO best" soll auch ohne fertiges best.pt benutzbar bleiben.
        # Deshalb fällt die UI automatisch auf eine vorhandene Gewichtsdatei
        # zurück, statt den Nutzer mit einem Dateifehler zu blockieren.
        for candidate_path in ("models/bestv11n_15ep.pt", "models/bestv8n_15ep.pt"):
            candidate = resolve_project_path(candidate_path)
            if candidate.exists():
                best_default = candidate
                break

    # Die sichtbaren Button-/Dropdown-Namen werden hier direkt auf konkrete
    # Modellpfade gemappt, damit die Auswahl eindeutig und nachvollziehbar ist.
    return {
        _MODEL_LABEL_V8: str(resolve_project_path("models/bestv8n_15ep.pt")),
        _MODEL_LABEL_V11: str(resolve_project_path("models/bestv11n_15ep.pt")),
        _MODEL_LABEL_BEST: str(best_default),
    }


def get_pipeline(
    detector_type_setting: str,
    yolo_weights_path: str,
    deepfashion2_root: str,
    detector_mode: str,
) -> InferencePipeline:
    # Pro Auswahl in der Oberfläche wird eine passende Pipeline erzeugt, damit
    # Mock- und YOLO-Modus dieselbe UI verwenden können.
    settings = Settings(
        detector_type=detector_type_setting,
        yolo_weights_path=str(resolve_project_path(yolo_weights_path)),
        deepfashion2_root=str(resolve_project_path(deepfashion2_root)),
    )
    return InferencePipeline(settings=settings, detector_type=detector_mode)


def predict_and_render_with_ultralytics(detector: object, image: Image.Image) -> tuple[list[dict[str, object]], Image.Image]:
    # Für den YOLO-Einzelbildtest wird direkt die Ultralytics-Zeichenroutine genutzt
    if hasattr(detector, "predict_and_render"):
        predictions, result_image = detector.predict_and_render(image)
        return predictions, result_image

    if hasattr(detector, "_predict_results") and hasattr(detector, "_map_predictions"):
        # Falls ein älterer Detector ohne Komfortmethode geladen ist, wird
        # über interne Hilfsfunktionen dennoch derselbe Renderpfad genutzt.
        results = detector._predict_results(image)
        predictions = detector._map_predictions(results)
    elif hasattr(detector, "model"):
        results = detector.model.predict(image, verbose=False)
        predictions = detector.predict(image)
    else:
        raise AttributeError("YOLO-Detector hat keinen nutzbaren Ultralytics-Zugriff.")

    if not results:
        raise ValueError("YOLO hat keine Ergebnisse zum Rendern geliefert.")

    plotted = results[0].plot()
    return predictions, Image.fromarray(plotted[..., ::-1])


def get_matchtester(dataset_root: str, weights_path: str, output_root: str) -> DeepFashion2MatchTester:
    # Der Matchtester wird gecacht, weil Modell- und Datensatzinitialisierung
    # vergleichsweise aufwendig sein kann.
    return DeepFashion2MatchTester(
        dataset_root=str(resolve_project_path(dataset_root)),
        weights_path=str(resolve_project_path(weights_path)),
        output_root=str(resolve_project_path(output_root)),
    )


def ensure_test_split_ready_matchtester(
    tester: DeepFashion2MatchTester,
    dataset_root: str,
    weights_path: str,
    output_root: str,
) -> DeepFashion2MatchTester:
    # Falls im laufenden Streamlit-Prozess doch noch ein älteres Tester-Objekt
    # ohne den neuen Testsplit-Pfad vorhanden ist, wird hier sofort eine frische
    # Instanz aufgebaut. Wenn nötig, wird sogar das Modul neu geladen.
    if hasattr(tester, "run_prediction_only") and hasattr(tester, "run_prediction_only_batch"):
        return tester

    refreshed_matchtest_module = importlib.reload(sys.modules["src.testing.matchtest"])
    refreshed_matchtester_class = refreshed_matchtest_module.DeepFashion2MatchTester
    return refreshed_matchtester_class(
        dataset_root=str(resolve_project_path(dataset_root)),
        weights_path=str(resolve_project_path(weights_path)),
        output_root=str(resolve_project_path(output_root)),
    )


def main() -> None:
    # Die Oberfläche trennt normalen Bildtest und Matchtest in zwei Tabs,
    # damit beide unabhängig bleiben.
    st.set_page_config(page_title="Kleidungs-Erkennung", layout="wide")
    st.title("Kleidungs-Erkennung (Schulprojekt)")
    st.caption("Demo-App für Einzelbildtest und Matchtests")

    settings = Settings()
    image_tab, match_tab = st.tabs(["Bildtest", "Matchtest"])

    with image_tab:
        render_image_test(settings)

    with match_tab:
        render_matchtest_page(settings)


def render_image_test(settings: Settings) -> None:
    # Dieser Bereich ist der einfachste Einstieg: Bild hochladen, Detector
    # auswählen, Ergebnisbild und Rohvorhersagen ansehen.
    st.subheader("Einzelbild testen")
    detector_mode = st.selectbox(
        "Detector",
        options=["mock", "yolo"],
        index=0 if settings.detector_type == "mock" else 1,
        key="detector_mode",
    )
    available_model_names = get_available_model_names_from_settings(settings)
    active_model_name = getattr(settings, "active_yolo_model", _MODEL_LABEL_V8)
    default_model_name = active_model_name if active_model_name in available_model_names else available_model_names[0]
    selected_model_name = default_model_name
    if detector_mode == "yolo":
        selected_model_name = st.selectbox(
            "YOLO-Modell",
            options=available_model_names,
            index=available_model_names.index(default_model_name),
            key="image_model_name",
        )
        weights_path = resolve_project_path(get_weights_path_from_settings(settings, selected_model_name))
        st.caption(f"YOLO-Gewichte: {weights_path}")
        if not weights_path.exists():
            st.error(f"Gewichtedatei fehlt: {weights_path}")
            return

    try:
        pipeline = get_pipeline(
            settings.detector_type,
            get_weights_path_from_settings(settings, selected_model_name),
            settings.deepfashion2_root,
            detector_mode,
        )
    except (FileNotFoundError, ImportError, ValueError) as error:
        st.error(f"Detector konnte nicht geladen werden: {error}")
        return

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file is None:
        st.info("Bitte ein Bild auswählen.")
        return

    image = load_uploaded_image(uploaded_file)
    if image is None:
        return

    try:
        if detector_mode == "yolo":
            predictions, result_image = predict_and_render_with_ultralytics(pipeline.detector, image)
        else:
            # Für den Mock-Modus bleibt die projektinterne Visualisierung aktiv.
            predictions = pipeline.predict(image)
            result_image = draw_predictions(image, predictions)
    except Exception as error:
        st.error(f"Vorhersage fehlgeschlagen: {error}")
        return

    st.image(result_image, caption="Vorhersage mit Bounding Boxes", width=_STREAMLIT_IMAGE_WIDTH)
    st.markdown("**Vorhersagen**")
    st.dataframe(predictions, use_container_width=True, hide_index=True)


def render_matchtest_page(settings: Settings) -> None:
    # Für den Matchtest wird ein sinnvoller DeepFashion2-Pfad automatisch
    # vorgeschlagen, kann aber jederzeit manuell überschrieben werden.
    detected_dataset_root = str(resolve_project_path(settings.deepfashion2_root))
    current_dataset_root = st.session_state.get("match_dataset_root")
    if current_dataset_root is None or (
        not discover_available_splits_for_ui(current_dataset_root)
        and discover_available_splits_for_ui(detected_dataset_root)
    ):
        st.session_state["match_dataset_root"] = detected_dataset_root

    dataset_root_value = st.session_state.get("match_dataset_root", detected_dataset_root)
    available_model_names = get_available_model_names_from_settings(settings)
    active_model_name = getattr(settings, "active_yolo_model", _MODEL_LABEL_V8)
    default_model_name = active_model_name if active_model_name in available_model_names else available_model_names[0]
    selected_match_model = st.session_state.get("match_model_name", default_model_name)
    if selected_match_model not in available_model_names:
        selected_match_model = default_model_name
        st.session_state["match_model_name"] = selected_match_model

    weights_path_value = st.session_state.get(
        "match_weights_path",
        get_weights_path_from_settings(settings, selected_match_model),
    )
    output_root_value = st.session_state.get("match_output_root", "outputs/matchtests")
    st.subheader("Matchtest")
    st.write("Hier könnt ihr Einzelbilder oder einen kleinen Batch direkt gegen Ground Truth vergleichen.")
    show_dataset_root_hint(dataset_root_value)
    available_splits = discover_available_splits_for_ui(dataset_root_value)
    split_options = available_splits or ["validation", "train", "test"]

    with st.form("matchtest_form"):
        # Alle Einstellungen liegen in einem Formular, damit die eigentliche
        # Ausführung erst nach dem bewussten Klick auf "starten" beginnt.
        left, right = st.columns(2, gap="large")
        with left:
            dataset_root = st.text_input("DeepFashion2 Pfad", value=dataset_root_value, key="match_dataset_root")
            model_name = st.selectbox(
                "YOLO-Modell",
                options=available_model_names,
                index=available_model_names.index(selected_match_model),
                key="match_model_name",
            )
            weights_path = get_weights_path_from_settings(settings, model_name)
            st.caption(f"YOLO-Gewichte: {resolve_project_path(weights_path)}")
            output_root = st.text_input("Output Ordner", value=output_root_value, key="match_output_root")
            split = st.selectbox("Split", options=split_options, index=0)
            if split == "test":
                st.caption("Im Test-Split werden nur YOLO-Vorhersagen gezeigt, weil keine Ground-Truth-Boxen vorhanden sind.")
        with right:
            mode = st.radio("Modus", options=["batch", "single"], horizontal=True)
            limit = st.slider("Batch-Größe", min_value=1, max_value=50, value=12)
            confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            iou_threshold = st.slider("IoU Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
            annotation_stem = st.text_input("Bild-/Annotation-Stem (nur single)", value="")

        submitted = st.form_submit_button("Matchtest starten", use_container_width=True)

    if not submitted:
        st.info("Werte setzen und dann den Matchtest starten.")
        return

    try:
        tester = get_matchtester(dataset_root, weights_path, output_root)
        if split == "test":
            tester = ensure_test_split_ready_matchtester(tester, dataset_root, weights_path, output_root)
    except (FileNotFoundError, ImportError, ValueError) as error:
        st.error(f"Matchtester konnte nicht vorbereitet werden: {error}")
        return

    try:
        with st.spinner("Matchtest läuft..."):
            # Der Test-Split hat keine Ground Truth. Deshalb gibt es dort nur
            # Vorhersagebilder, aber keine Recall-/Precision-Bewertung.
            if split == "test" and mode == "single":
                result = tester.run_prediction_only(
                    split=split,
                    image_stem=annotation_stem.strip() or None,
                    confidence_threshold=confidence_threshold,
                )
                render_single_prediction_result(result)
            elif split == "test":
                result = tester.run_prediction_only_batch(
                    split=split,
                    limit=limit,
                    confidence_threshold=confidence_threshold,
                )
                render_batch_prediction_result(result)
            elif mode == "single":
                # Der Einzelmodus erzeugt ein Vergleichsbild für genau eine Annotation.
                result = tester.run(
                    split=split,
                    annotation_stem=annotation_stem.strip() or None,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                )
                render_single_match_result(result)
            else:
                # Der Batchmodus erzeugt mehrere Vergleichsbilder plus Sammelberichte.
                result = tester.run_batch(
                    split=split,
                    limit=limit,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                )
                render_batch_match_result(result)
    except Exception as error:
        st.error(f"Matchtest fehlgeschlagen: {error}")
        st.info(
            "Tipp: Im Feld 'DeepFashion2 Pfad' den echten Datensatz-Ordner eintragen, "
            "zum Beispiel 'F:\\Schulprojekt KI\\DeepFashion2'."
        )
        if available_splits:
            st.caption(f"Am aktuellen Pfad erkannte Splits: {', '.join(available_splits)}")


def render_single_match_result(result: dict[str, object]) -> None:
    # Die Einzelansicht zeigt zuerst die wichtigsten Kennzahlen und darunter
    # das erzeugte Vergleichsbild.
    st.success("Einzelner Matchtest abgeschlossen.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("GT Boxen", int(result["ground_truth_count"]))
    metric_columns[1].metric("Vorhersagen", int(result["prediction_count"]))
    metric_columns[2].metric("Matches", int(result["matches"]))
    metric_columns[3].metric("Recall", format_percentage(float(result["recall"])))

    image_path = Path(str(result["output_image_path"]))
    if image_path.exists():
        with Image.open(image_path) as result_image:
            st.image(result_image.copy(), caption=image_path.name, width=_STREAMLIT_IMAGE_WIDTH)

    st.markdown("**Match Records**")
    st.dataframe(result["match_records"], use_container_width=True, hide_index=True)

    summary_path = Path(str(result["summary_path"]))
    if summary_path.exists():
        st.download_button(
            "JSON-Zusammenfassung herunterladen",
            data=summary_path.read_text(encoding="utf-8"),
            file_name=summary_path.name,
            mime="application/json",
        )


def render_batch_match_result(result: dict[str, object]) -> None:
    # Der Batchbereich fasst Gesamtmetriken, Reportbilder und Beispielergebnisse
    # in einer scrollbaren Übersicht zusammen.
    st.success("Batch-Matchtest abgeschlossen.")

    top_metrics = st.columns(6)
    top_metrics[0].metric("Bilder", int(result["processed_images"]))
    top_metrics[1].metric("GT Boxen", int(result["ground_truth_count"]))
    top_metrics[2].metric("Vorhersagen", int(result["prediction_count"]))
    top_metrics[3].metric("Matches", int(result["match_count"]))
    top_metrics[4].metric("Recall", format_percentage(float(result["recall"])))
    top_metrics[5].metric("Precision", format_percentage(float(result["precision"])))

    report_path = Path(str(result["report_image_path"]))
    if report_path.exists():
        with Image.open(report_path) as report_image:
            st.image(report_image.copy(), caption="Batch-Report", width=_STREAMLIT_IMAGE_WIDTH)

    category_rows = list(result.get("categories", []))
    if category_rows:
        st.markdown("**Kategorien**")
        st.dataframe(category_rows, use_container_width=True, hide_index=True)

    image_rows = list(result.get("images", []))
    if image_rows:
        st.markdown("**Beispielbilder**")
        for index, row in enumerate(image_rows[:6]):
            # Es werden nur einige Beispielbilder gezeigt, damit die Seite
            # bei größeren Läufen übersichtlich bleibt.
            image_path = Path(str(row["output_image_path"]))
            if image_path.exists():
                with Image.open(image_path) as sample_image:
                    st.image(sample_image.copy(), caption=str(row["annotation_stem"]), width=_STREAMLIT_IMAGE_WIDTH)
            st.caption(
                f"Recall {format_percentage(float(row['recall']))} | "
                f"Precision {format_percentage(float(row['precision']))}"
            )

    files_column, json_column = st.columns(2, gap="large")
    with files_column:
        st.markdown("**Report-Dateien**")
        st.write(f"PNG: `{result['report_image_path']}`")
        st.write(f"HTML: `{result['report_html_path']}`")
        st.write(f"JSON: `{result['summary_json_path']}`")
    with json_column:
        summary_path = Path(str(result["summary_json_path"]))
        if summary_path.exists():
            st.download_button(
                "Batch-JSON herunterladen",
                data=summary_path.read_text(encoding="utf-8"),
                file_name=summary_path.name,
                mime="application/json",
            )
            with st.expander("JSON-Vorschau"):
                st.code(summary_path.read_text(encoding="utf-8"), language="json")


def render_single_prediction_result(result: dict[str, object]) -> None:
    # Für den Test-Split ohne Ground Truth werden nur Vorhersagen und das
    # gerenderte Ergebnisbild gezeigt.
    st.success("Testsplit-Vorhersage abgeschlossen.")

    metric_columns = st.columns(2)
    metric_columns[0].metric("Split", str(result["split"]))
    metric_columns[1].metric("Vorhersagen", int(result["prediction_count"]))

    image_path = Path(str(result["output_image_path"]))
    if image_path.exists():
        with Image.open(image_path) as result_image:
            st.image(result_image.copy(), caption=image_path.name, width=_STREAMLIT_IMAGE_WIDTH)

    predictions = list(result.get("predictions", []))
    if predictions:
        st.markdown("**Vorhersagen**")
        st.dataframe(predictions, use_container_width=True, hide_index=True)

    summary_path = Path(str(result["summary_path"]))
    if summary_path.exists():
        st.download_button(
            "JSON-Zusammenfassung herunterladen",
            data=summary_path.read_text(encoding="utf-8"),
            file_name=summary_path.name,
            mime="application/json",
        )


def render_batch_prediction_result(result: dict[str, object]) -> None:
    # Auch im Batchmodus für den Test-Split werden nur Bildanzahl und
    # Vorhersagemengen gezeigt, nicht jedoch Matchmetriken.
    st.success("Testsplit-Batchvorhersage abgeschlossen.")

    top_metrics = st.columns(3)
    top_metrics[0].metric("Split", str(result["split"]))
    top_metrics[1].metric("Bilder", int(result["processed_images"]))
    top_metrics[2].metric("Vorhersagen", int(result["prediction_count"]))

    image_rows = list(result.get("images", []))
    if image_rows:
        st.markdown("**Beispielbilder**")
        for row in image_rows[:6]:
            image_path = Path(str(row["output_image_path"]))
            if image_path.exists():
                with Image.open(image_path) as sample_image:
                    st.image(sample_image.copy(), caption=str(row["image_stem"]), width=_STREAMLIT_IMAGE_WIDTH)
            st.caption(f"Vorhersagen: {int(row['prediction_count'])}")

    summary_path = Path(str(result["summary_json_path"]))
    if summary_path.exists():
        st.download_button(
            "Batch-JSON herunterladen",
            data=summary_path.read_text(encoding="utf-8"),
            file_name=summary_path.name,
            mime="application/json",
        )
        with st.expander("JSON-Vorschau"):
            st.code(summary_path.read_text(encoding="utf-8"), language="json")


def load_uploaded_image(uploaded_file: object) -> Image | None:
    # Hochgeladene Bilder werden direkt in ein einheitliches RGB-Format gebracht.
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        return ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError:
        st.error("Dateiformat nicht lesbar. Bitte JPG oder PNG verwenden.")
    except OSError as error:
        st.error(f"Bild konnte nicht geöffnet werden: {error}")
    return None


def format_percentage(value: float) -> str:
    # Einheitliche Prozentdarstellung für Metriken in der Oberfläche.
    return f"{value * 100:.1f}%"


def show_dataset_root_hint(dataset_root: str) -> None:
    # Nutzer bekommen sofort Rückmeldung, ob der aktuelle Datensatzpfad
    # plausibel aussieht oder noch korrigiert werden sollte.
    dataset_path = resolve_project_path(dataset_root)
    available_splits = discover_available_splits_for_ui(dataset_path)
    if available_splits:
        st.success(f"DeepFashion2 erkannt unter: {dataset_path}")
        st.caption(f"Verfügbare Splits: {', '.join(available_splits)}")
        return

    st.warning(
        "DeepFashion2 wurde am aktuellen Pfad noch nicht erkannt. "
        "Erwartet wird zum Beispiel '...\\DeepFashion2\\validation\\image' plus 'annos' "
        "oder entsprechend für train/test."
    )
    st.caption(f"Aktueller Pfad: {dataset_path}")


def looks_like_deepfashion2_root(dataset_root: Path) -> bool:
    # Kleine Hilfsfunktion für spätere Prüfungen oder Erweiterungen.
    return bool(discover_available_splits_for_ui(dataset_root))


def discover_available_splits_for_ui(dataset_root: str | Path) -> list[str]:
    # Die UI erkennt automatisch vorhandene Splits und kann daraus passende
    # Auswahloptionen im Formular bauen.
    root = resolve_project_path(dataset_root)
    if _is_split_root_for_ui(root):
        return [root.name]

    available: list[str] = []
    for split in ("validation", "train", "test"):
        direct_root = root / split
        nested_root = direct_root / split
        if (
            _is_split_root_for_ui(direct_root)
            or _is_split_root_for_ui(nested_root)
            or _is_image_only_split_root_for_ui(direct_root)
            or _is_image_only_split_root_for_ui(nested_root)
        ):
            available.append(split)
    return available


def _is_split_root_for_ui(path: Path) -> bool:
    # Dieselbe Grundregel wie im Backend: Bild- und Annotationsordner müssen existieren.
    return (path / "image").exists() and (path / "annos").exists()


def _is_image_only_split_root_for_ui(path: Path) -> bool:
    # Für den reinen Vorhersagemodus reicht beim Test-Split bereits ein Bildordner.
    return (path / "image").exists()


if __name__ == "__main__":
    main()
