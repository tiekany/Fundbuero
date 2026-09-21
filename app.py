# ============================================================
# FUNDBÜRO – KATHARINEUM ZU LÜBECK
# Streamlit Lost & Found App
# ============================================================

# ------------------------------------------------------------
# WICHTIG:
# TF_USE_LEGACY_KERAS muss VOR dem TensorFlow-Import gesetzt
# werden, damit ältere Teachable-Machine-H5-Modelle besser
# mit TensorFlow 2.16 funktionieren.
# ------------------------------------------------------------

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# 1. GRUNDEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="Fundbüro",
    page_icon="assets/app_icon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)


BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "modell"
IMAGE_DIR = BASE_DIR / "bilder"
DATA_DIR = BASE_DIR / "daten"

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

DATABASE_PATH = DATA_DIR / "fundbuero.db"
FIXED_MODEL_PATH = DATA_DIR / "keras_model_fixed.h5"

LOGO_PATH = ASSETS_DIR / "fundbuero_logo.png"
FOOTER_LOGO_PATH = ASSETS_DIR / "footer_logo.png"
APP_ICON_PATH = ASSETS_DIR / "app_icon.png"
PROFILE_ICON_PATH = ASSETS_DIR / "profile.png"
UPLOAD_ICON_PATH = ASSETS_DIR / "upload.png"
SEARCH_ICON_PATH = ASSETS_DIR / "search.png"
BACK_ICON_PATH = ASSETS_DIR / "back.png"

IMAGE_SIZE = (224, 224)


# Ordner automatisch erstellen
ASSETS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. DESIGN / CSS
# ============================================================
#
# Hier wird KEIN sichtbarer HTML-Inhalt erzeugt.
# Dieser Block dient ausschließlich dem Styling.
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Allgemein
       ------------------------------------------------------- */

    .stApp {
        background: #ffffff;
    }

    .main .block-container {
        max-width: 460px;
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Alles etwas kompakter für Handy */
    @media (max-width: 600px) {

        .main .block-container {
            max-width: 100%;
            padding-left: 12px;
            padding-right: 12px;
            padding-top: 10px;
        }

        h1 {
            font-size: 28px !important;
        }

        h2 {
            font-size: 22px !important;
        }

        h3 {
            font-size: 18px !important;
        }
    }


    /* -------------------------------------------------------
       Buttons
       ------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        min-height: 54px;
        border-radius: 4px;
        border: 3px solid #111111;
        background: white;
        color: #111111;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        border-color: #111111;
        color: #111111;
        background: #f3f3f3;
    }

    .stButton > button:focus {
        border-color: #111111;
        color: #111111;
        box-shadow: none;
    }


    /* -------------------------------------------------------
       Textfelder
       ------------------------------------------------------- */

    .stTextInput input {
        border: 3px solid #111111 !important;
        border-radius: 3px !important;
        color: #111111 !important;
        font-size: 16px !important;
    }

    .stTextInput label {
        font-weight: 700;
        color: #111111;
    }


    /* -------------------------------------------------------
       Selectbox
       ------------------------------------------------------- */

    .stSelectbox > div > div {
        border: 2px solid #111111 !important;
        border-radius: 3px !important;
    }


    /* -------------------------------------------------------
       Bilder
       ------------------------------------------------------- */

    img {
        object-fit: contain;
    }


    /* -------------------------------------------------------
       Success / Warnungen
       ------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 4px;
    }


    /* -------------------------------------------------------
       Footer
       ------------------------------------------------------- */

    .footer-text {
        text-align: center;
        font-weight: 800;
        font-size: 12px;
        margin-top: 12px;
        margin-bottom: 4px;
    }


    /* -------------------------------------------------------
       Trennlinie
       ------------------------------------------------------- */

    hr {
        border-color: #111111;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

if "search_term" not in st.session_state:
    st.session_state.search_term = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# 4. HILFSFUNKTIONEN
# ============================================================

def go_to(page):
    """Wechselt die aktuelle Seite."""
    st.session_state.page = page
    st.rerun()


def save_uploaded_image(uploaded_file):
    """
    Speichert ein hochgeladenes Bild im Bilder-Ordner
    und gibt den Dateipfad zurück.
    """

    if uploaded_file is None:
        return None

    try:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")

        filename = f"{uuid.uuid4().hex}.jpg"
        path = IMAGE_DIR / filename

        image.save(path, quality=92)

        return str(path)

    except Exception as error:
        st.error(f"Bild konnte nicht gespeichert werden: {error}")
        return None


def get_image(path):
    """Öffnet ein Bild sicher."""

    try:
        if path and Path(path).exists():
            return Image.open(path)
    except Exception:
        pass

    return None


# ============================================================
# 5. DATENBANK
# ============================================================

def get_connection():
    """Verbindung zur SQLite-Datenbank."""

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fundstuecke (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            name TEXT,
            kategorie TEXT,
            farbe TEXT,
            gefunden_am TEXT,
            fundort TEXT,
            notizen TEXT,
            aktueller_standort TEXT,
            zugeordnet INTEGER DEFAULT 0,
            erstellt_am TEXT
        )
        """
    )

    connection.commit()
    connection.close()


initialize_database()


def add_item(
    image_path,
    name,
    kategorie,
    farbe,
    gefunden_am,
    fundort,
    notizen,
    aktueller_standort,
):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO fundstuecke (
            image_path,
            name,
            kategorie,
            farbe,
            gefunden_am,
            fundort,
            notizen,
            aktueller_standort,
            zugeordnet,
            erstellt_am
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            image_path,
            name,
            kategorie,
            farbe,
            gefunden_am,
            fundort,
            notizen,
            aktueller_standort,
            datetime.now().isoformat(),
        ),
    )

    connection.commit()

    item_id = cursor.lastrowid

    connection.close()

    return item_id


def get_all_items():
    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM fundstuecke
        ORDER BY id DESC
        """
    )

    items = cursor.fetchall()

    connection.close()

    return items


def get_item(item_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM fundstuecke
        WHERE id = ?
        """,
        (item_id,),
    )

    item = cursor.fetchone()

    connection.close()

    return item


def update_assigned(item_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE fundstuecke
        SET zugeordnet = 1
        WHERE id = ?
        """,
        (item_id,),
    )

    connection.commit()
    connection.close()


# ============================================================
# 6. LABELS DES KI-MODELLS
# ============================================================

def load_labels():

    if not LABELS_PATH.exists():
        return [
            "Klasse 1",
            "Klasse 2",
            "Klasse 3",
            "Klasse 4",
            "Klasse 5",
        ]

    try:

        labels = []

        with open(LABELS_PATH, "r", encoding="utf-8") as file:

            for line in file:
                line = line.strip()

                if line:
                    labels.append(line)

        if labels:
            return labels

    except Exception:
        pass

    return [
        "Klasse 1",
        "Klasse 2",
        "Klasse 3",
        "Klasse 4",
        "Klasse 5",
    ]


LABELS = load_labels()


# ============================================================
# 7. H5-MODELL REPARIEREN
# ============================================================

def repair_h5_model():

    """
    Repariert ältere Teachable-Machine-H5-Dateien.

    Problem:
    Ältere Modelle enthalten bei DepthwiseConv2D teilweise
    den Parameter "groups", den neuere Keras-Versionen
    nicht akzeptieren.
    """

    if not MODEL_PATH.exists():
        return None

    if FIXED_MODEL_PATH.exists():

        try:

            source_time = MODEL_PATH.stat().st_mtime
            fixed_time = FIXED_MODEL_PATH.stat().st_mtime

            if fixed_time >= source_time:
                return FIXED_MODEL_PATH

        except Exception:
            pass

    try:

        import h5py

        shutil.copy2(
            MODEL_PATH,
            FIXED_MODEL_PATH,
        )

        with h5py.File(FIXED_MODEL_PATH, "r+") as file:

            if "model_config" not in file.attrs:
                return FIXED_MODEL_PATH

            config = file.attrs["model_config"]

            if isinstance(config, bytes):
                config = config.decode("utf-8")

            config = json.loads(config)

            def clean_config(obj):

                if isinstance(obj, dict):

                    class_name = obj.get("class_name")

                    if class_name == "DepthwiseConv2D":

                        obj.get("config", {}).pop(
                            "groups",
                            None,
                        )

                    for value in obj.values():
                        clean_config(value)

                elif isinstance(obj, list):

                    for value in obj:
                        clean_config(value)

            clean_config(config)

            file.attrs["model_config"] = json.dumps(config)

        return FIXED_MODEL_PATH

    except Exception as error:

        st.warning(
            f"Das H5-Modell konnte nicht automatisch repariert werden: {error}"
        )

        return MODEL_PATH


# ============================================================
# 8. KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_ai_model():

    model_file = repair_h5_model()

    if model_file is None:
        return None

    try:

        model = tf.keras.models.load_model(
            model_file,
            compile=False,
        )

        return model

    except Exception as error:

        st.error(
            "Das KI-Modell konnte nicht geladen werden."
        )

        st.code(str(error))

        return None


# ============================================================
# 9. BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )

    image_data = tf.keras.utils.img_to_array(image)

    image_data = image_data.astype("float32")

    # Teachable Machine / MobileNet verwendet
    # normalerweise Werte zwischen -1 und 1.
    image_data = (image_data / 127.5) - 1.0

    image_data = tf.expand_dims(
        image_data,
        axis=0,
    )

    return image_data


# ============================================================
# 10. EINZELNE LAYER AUSFÜHREN
# ============================================================

def run_layers(model_part, tensor):

    """
    Führt die Layer eines verschachtelten Modells
    einzeln aus.

    Dadurch vermeiden wir den Fehler:

    "expects 1 named input ... received 2 input tensors"
    """

    if model_part is None:
        return tensor

    layers = getattr(
        model_part,
        "layers",
        [],
    )

    for layer in layers:

        # InputLayer nicht noch einmal ausführen
        if isinstance(
            layer,
            tf.keras.layers.InputLayer,
        ):
            continue

        tensor = layer(
            tensor,
            training=False,
        )

    return tensor


# ============================================================
# 11. KI-VORHERSAGE
# ============================================================

def predict_image(image):

    model = load_ai_model()

    if model is None:
        return None

    try:

        tensor = prepare_image(image)

        outer_layers = list(
            getattr(model, "layers", [])
        )

        # ----------------------------------------------------
        # Fall A:
        # Das Teachable-Machine-Modell besteht aus:
        #
        # Input
        # sequential_1
        # sequential_3
        # ----------------------------------------------------

        feature_model = None
        classifier_model = None

        for layer in outer_layers:

            layer_name = getattr(
                layer,
                "name",
                "",
            )

            if layer_name == "sequential_1":
                feature_model = layer

            elif layer_name == "sequential_3":
                classifier_model = layer

        # ----------------------------------------------------
        # Wenn die Namen nicht stimmen, versuchen wir die
        # verschachtelten Modelle automatisch zu finden.
        # ----------------------------------------------------

        if feature_model is None or classifier_model is None:

            nested_models = []

            for layer in outer_layers:

                if hasattr(layer, "layers"):

                    inner_layers = getattr(
                        layer,
                        "layers",
                        [],
                    )

                    if len(inner_layers) >= 2:
                        nested_models.append(layer)

            if len(nested_models) >= 2:

                feature_model = nested_models[0]
                classifier_model = nested_models[-1]

        # ----------------------------------------------------
        # Standardfall
        # ----------------------------------------------------

        if feature_model is not None:

            tensor = run_layers(
                feature_model,
                tensor,
            )

            if classifier_model is not None:

                tensor = run_layers(
                    classifier_model,
                    tensor,
                )

            else:

                # Falls kein Klassifikator gefunden wurde,
                # versuchen wir die restlichen Layer des
                # äußeren Modells.
                for layer in outer_layers:

                    if layer is feature_model:
                        continue

                    if isinstance(
                        layer,
                        tf.keras.layers.InputLayer,
                    ):
                        continue

                    if hasattr(layer, "layers"):
                        continue

                    tensor = layer(
                        tensor,
                        training=False,
                    )

        else:

            # ------------------------------------------------
            # Fallback:
            # äußere Layer einzeln ausführen
            # ------------------------------------------------

            tensor = run_layers(
                model,
                tensor,
            )

        # ----------------------------------------------------
        # Ergebnis
        # ----------------------------------------------------

        probabilities = tensor.numpy()

        probabilities = probabilities.reshape(-1)

        # Falls das Modell keine Wahrscheinlichkeiten liefert
        # und stattdessen Logits liefert.
        total = float(probabilities.sum())

        if (
            total < 0.99
            or total > 1.01
            or any(probabilities < 0)
        ):

            probabilities = tf.nn.softmax(
                probabilities
            ).numpy()

        ranking = sorted(
            enumerate(probabilities),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for index, probability in ranking:

            if index < len(LABELS):
                label = LABELS[index]
            else:
                label = f"Klasse {index + 1}"

            results.append(
                {
                    "label": label,
                    "confidence": float(probability),
                }
            )

        return results

    except Exception as error:

        st.error(
            "KI-Erkennung fehlgeschlagen."
        )

        with st.expander("Technische Fehlermeldung"):

            st.code(
                str(error)
            )

        return None


# ============================================================
# 12. HEADER
# ============================================================

def show_header(
    show_back=False,
    show_profile=True,
):

    # --------------------------------------------------------
    # obere Zeile
    # --------------------------------------------------------

    if show_back:

        left, center, right = st.columns(
            [1, 6, 1]
        )

        with left:

            if BACK_ICON_PATH.exists():

                if st.button(
                    "←",
                    key="back_top",
                    width="content",
                ):
                    go_to("start")

            else:

                if st.button(
                    "←",
                    key="back_top",
                    width="content",
                ):
                    go_to("start")

        with center:

            if LOGO_PATH.exists():

                st.image(
                    LOGO_PATH,
                    width="content",
                )

            else:

                st.title("FUNDBÜRO")

        with right:

            if (
                show_profile
                and PROFILE_ICON_PATH.exists()
            ):

                st.image(
                    PROFILE_ICON_PATH,
                    width="content",
                )

    else:

        left, center, right = st.columns(
            [1, 6, 1]
        )

        with center:

            if LOGO_PATH.exists():

                st.image(
                    LOGO_PATH,
                    width="content",
                )

            else:

                st.title("FUNDBÜRO")

        with right:

            if (
                show_profile
                and PROFILE_ICON_PATH.exists()
            ):

                st.image(
                    PROFILE_ICON_PATH,
                    width="content",
                )


# ============================================================
# 13. FOOTER
# ============================================================

def show_footer():

    st.divider()

    if FOOTER_LOGO_PATH.exists():

        st.image(
            FOOTER_LOGO_PATH,
            width="content",
        )

    else:

        st.markdown(
            "**TU ES**"
        )

        st.markdown(
            "**KATHARINEUM ZU LÜBECK**"
        )


# ============================================================
# 14. STARTSEITE
# ============================================================

def page_start():

    show_header()

    st.write("")

    # --------------------------------------------------------
    # Foto hochladen
    # --------------------------------------------------------

    if UPLOAD_ICON_PATH.exists():
        st.image(
            UPLOAD_ICON_PATH,
            width=35,
        )

    if st.button(
        "FOTO HOCHLADEN",
        key="upload_button",
        width="stretch",
    ):

        st.session_state.page = "upload"
        st.rerun()

    st.write("")

    # --------------------------------------------------------
    # Kleidung suchen
    # --------------------------------------------------------

    if SEARCH_ICON_PATH.exists():
        st.image(
            SEARCH_ICON_PATH,
            width=35,
        )

    if st.button(
        "KLEIDUNGSSTÜCK SUCHEN",
        key="search_button",
        width="stretch",
    ):

        st.session_state.page = "search"
        st.rerun()

    st.write("")

    st.subheader("LETZTE FUNDSTÜCKE")

    items = get_all_items()

    if not items:

        st.info(
            "Noch keine Fundstücke vorhanden."
        )

    else:

        # Die letzten sechs Fundstücke
        recent_items = items[:6]

        columns = st.columns(3)

        for index, item in enumerate(
            recent_items
        ):

            column = columns[
                index % 3
            ]

            with column:

                image = get_image(
                    item["image_path"]
                )

                if image:

                    st.image(
                        image,
                        width="stretch",
                    )

                if st.button(
                    "Ansehen",
                    key=f"recent_{item['id']}",
                    width="stretch",
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go_to("detail")


    show_footer()


# ============================================================
# 15. UPLOAD-SEITE
# ============================================================

def page_upload():

    show_header(
        show_back=True
    )

    st.subheader(
        "FUNDSTÜCK AUFNEHMEN"
    )

    st.write(
        "Fotografiere oder lade ein Bild des Fundstücks hoch."
    )

    # --------------------------------------------------------
    # Kamera
    # --------------------------------------------------------

    camera_image = st.camera_input(
        "Foto aufnehmen"
    )

    # --------------------------------------------------------
    # Datei
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Oder Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    selected_image = None

    if camera_image is not None:

        selected_image = camera_image

    elif uploaded_file is not None:

        selected_image = uploaded_file


    # --------------------------------------------------------
    # Wenn ein Bild ausgewählt wurde
    # --------------------------------------------------------

    if selected_image is not None:

        image = Image.open(
            selected_image
        ).convert("RGB")

        st.subheader(
            "Vorschau"
        )

        st.image(
            image,
            width="stretch",
        )

        # ----------------------------------------------------
        # KI-Erkennung
        # ----------------------------------------------------

        st.subheader(
            "KI-Erkennung"
        )

        with st.spinner(
            "Fundstück wird erkannt..."
        ):

            results = predict_image(
                image
            )

        if results:

            best = results[0]

            confidence = (
                best["confidence"] * 100
            )

            st.write(
                f"**Erkannt:** {best['label']}"
            )

            st.write(
                f"**Sicherheit:** {confidence:.1f} %"
            )

            # weitere Ergebnisse
            with st.expander(
                "Weitere KI-Ergebnisse"
            ):

                for result in results:

                    percentage = (
                        result["confidence"]
                        * 100
                    )

                    st.write(
                        f"{result['label']}: "
                        f"{percentage:.1f} %"
                    )

        # ----------------------------------------------------
        # Daten zum Fundstück
        # ----------------------------------------------------

        st.subheader(
            "Informationen zum Fundstück"
        )

        detected_name = ""

        if results:
            detected_name = results[0][
                "label"
            ]

        name = st.text_input(
            "Bezeichnung",
            value=detected_name,
        )

        category_options = [
            "",
            "Kleidung",
            "Flasche",
            "Tasche",
            "Schulsachen",
            "Elektronik",
            "Sonstiges",
        ]

        category = st.selectbox(
            "Kategorie",
            category_options,
        )

        color = st.text_input(
            "Farbe",
            placeholder="z. B. schwarz",
        )

        found_date = st.date_input(
            "Gefunden am"
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Obere Turnhalle",
        )

        notes = st.text_area(
            "Notizen",
            placeholder=(
                "Weitere Informationen zum Fundstück..."
            ),
        )

        current_location = st.text_input(
            "Aktueller Standort",
            value="Fundkiste",
        )

        st.write("")

        if st.button(
            "FUNDSTÜCK SPEICHERN",
            key="save_item",
            width="stretch",
        ):

            image_path = save_uploaded_image(
                selected_image
            )

            if image_path is None:

                st.error(
                    "Das Bild konnte nicht gespeichert werden."
                )

            else:

                item_id = add_item(
                    image_path=image_path,
                    name=name,
                    kategorie=category,
                    farbe=color,
                    gefunden_am=found_date.isoformat(),
                    fundort=location,
                    notizen=notes,
                    aktueller_standort=current_location,
                )

                st.success(
                    "Fundstück wurde gespeichert."
                )

                st.session_state.selected_item = (
                    item_id
                )

                st.session_state.page = (
                    "detail"
                )

                st.rerun()


    show_footer()


# ============================================================
# 16. SUCHSEITE
# ============================================================

def page_search():

    show_header(
        show_back=True
    )

    st.subheader(
        "FUNDSTÜCK SUCHEN"
    )

    search_term = st.text_input(
        "Suche",
        value=st.session_state.search_term,
        placeholder="(z.B.:) Flaschen",
    )

    st.session_state.search_term = (
        search_term
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    with st.expander(
        "Filter"
    ):

        color_filter = st.text_input(
            "Farbe",
            placeholder="z. B. schwarz",
        )

        category_filter = st.selectbox(
            "Kategorie",
            [
                "",
                "Kleidung",
                "Flasche",
                "Tasche",
                "Schulsachen",
                "Elektronik",
                "Sonstiges",
            ],
        )

    # --------------------------------------------------------
    # Ergebnisse
    # --------------------------------------------------------

    items = get_all_items()

    filtered_items = []

    search_lower = (
        search_term.strip().lower()
    )

    color_lower = (
        color_filter.strip().lower()
    )

    for item in items:

        searchable = " ".join(
            [
                item["name"] or "",
                item["kategorie"] or "",
                item["farbe"] or "",
                item["fundort"] or "",
                item["notizen"] or "",
            ]
        ).lower()

        # Textsuche
        if search_lower:

            if search_lower not in searchable:
                continue

        # Farbfilter
        if color_lower:

            item_color = (
                item["farbe"] or ""
            ).lower()

            if color_lower not in item_color:
                continue

        # Kategorienfilter
        if category_filter:

            if (
                item["kategorie"]
                != category_filter
            ):
                continue

        filtered_items.append(item)


    # --------------------------------------------------------
    # Überschrift
    # --------------------------------------------------------

    if search_term:

        st.subheader(
            f'ERGEBNISSE FÜR „{search_term.upper()}“'
        )

    else:

        st.subheader(
            "ALLE FUNDSTÜCKE"
        )


    if not filtered_items:

        st.info(
            "Keine passenden Fundstücke gefunden."
        )

    else:

        columns = st.columns(3)

        for index, item in enumerate(
            filtered_items
        ):

            with columns[
                index % 3
            ]:

                image = get_image(
                    item["image_path"]
                )

                if image:

                    st.image(
                        image,
                        width="stretch",
                    )

                label = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )

                st.caption(
                    label
                )

                if st.button(
                    "Ansehen",
                    key=f"search_{item['id']}",
                    width="stretch",
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go_to("detail")


    show_footer()


# ============================================================
# 17. DETAILSEITE
# ============================================================

def page_detail():

    show_header(
        show_back=True
    )

    item_id = (
        st.session_state.selected_item
    )

    if item_id is None:

        st.warning(
            "Kein Fundstück ausgewählt."
        )

        if st.button(
            "ZUR STARTSEITE",
            width="stretch",
        ):

            go_to("start")

        return


    item = get_item(
        item_id
    )

    if item is None:

        st.error(
            "Fundstück wurde nicht gefunden."
        )

        return


    # --------------------------------------------------------
    # Bild
    # --------------------------------------------------------

    image = get_image(
        item["image_path"]
    )

    if image:

        st.image(
            image,
            width="stretch",
        )


    # --------------------------------------------------------
    # Titel
    # --------------------------------------------------------

    title = (
        item["name"]
        or item["kategorie"]
        or "FUNDSTÜCK"
    )

    st.subheader(
        title.upper()
    )


    # --------------------------------------------------------
    # Informationen
    # --------------------------------------------------------

    st.write(
        "**GEFUNDEN AM:**"
    )

    st.write(
        item["gefunden_am"]
        or "-"
    )

    st.divider()

    st.write(
        "**FUNDORT:**"
    )

    st.write(
        item["fundort"]
        or "-"
    )

    st.divider()

    st.write(
        "**NOTIZEN:**"
    )

    st.write(
        item["notizen"]
        or "-"
    )

    st.divider()

    st.write(
        "**AKTUELLER STANDORT:**"
    )

    st.write(
        item["aktueller_standort"]
        or "-"
    )

    st.divider()

    if item["farbe"]:

        st.write(
            "**FARBE:**"
        )

        st.write(
            item["farbe"]
        )

        st.divider()


    # --------------------------------------------------------
    # Zuordnung
    # --------------------------------------------------------

    if item["zugeordnet"]:

        st.success(
            "FUNDSTÜCK ZUGEORDNET"
        )

    else:

        if st.button(
            "FUNDSTÜCK ZUORDNEN",
            key=f"assign_{item['id']}",
            width="stretch",
        ):

            update_assigned(
                item["id"]
            )

            st.success(
                "Fundstück wurde zugeordnet."
            )

            st.rerun()


    show_footer()


# ============================================================
# 18. SEITENSTEUERUNG
# ============================================================

if st.session_state.page == "start":

    page_start()

elif st.session_state.page == "upload":

    page_upload()

elif st.session_state.page == "search":

    page_search()

elif st.session_state.page == "detail":

    page_detail()

else:

    st.session_state.page = "start"

    page_start()
