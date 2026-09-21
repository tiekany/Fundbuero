import os
import re
import json
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# ============================================================
# KERAS / TENSORFLOW
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# SEITENEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon="assets/app_icon.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PFADE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "modell"
IMAGE_DIR = BASE_DIR / "bilder"
DATA_DIR = BASE_DIR / "daten"

ASSETS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# DATEIEN
# ============================================================

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

DATABASE_PATH = DATA_DIR / "fundbuero.db"
FIXED_MODEL_PATH = DATA_DIR / "keras_model_fixed.h5"

# Logos
LOGO_PATH = ASSETS_DIR / "fundbuero_logo.png"
FOOTER_LOGO_PATH = ASSETS_DIR / "footer_logo.png"
APP_ICON_PATH = ASSETS_DIR / "app_icon.png"

# Icons
PROFILE_ICON_PATH = ASSETS_DIR / "profile.png"
UPLOAD_ICON_PATH = ASSETS_DIR / "upload.png"
SEARCH_ICON_PATH = ASSETS_DIR / "search.png"
BACK_ICON_PATH = ASSETS_DIR / "back.png"

IMAGE_SIZE = (224, 224)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GRUNDLAYOUT
       -------------------------------------------------------- */

    .stApp {
        background: white;
    }

    .main .block-container {
        max-width: 450px;
        padding-top: 8px;
        padding-left: 10px;
        padding-right: 10px;
        padding-bottom: 5px;
    }


    /* --------------------------------------------------------
       STREAMLIT ELEMENTE
       -------------------------------------------------------- */

    header {
        visibility: hidden;
        height: 0;
    }

    footer {
        visibility: hidden;
        height: 0;
    }


    /* --------------------------------------------------------
       ÜBERSCHRIFTEN
       -------------------------------------------------------- */

    h1,
    h2,
    h3 {
        color: #111111 !important;
        font-weight: 900 !important;
    }

    h3 {
        font-size: 19px !important;
        margin-top: 8px !important;
        margin-bottom: 9px !important;
    }


    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button {
        width: 100% !important;

        min-height: 52px !important;

        border: 3px solid #111111 !important;
        border-radius: 2px !important;

        background: #ffffff !important;
        color: #111111 !important;

        font-size: 14px !important;
        font-weight: 900 !important;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #f3f3f3 !important;
        color: #111111 !important;
        border-color: #111111 !important;
    }

    .stButton > button:focus {
        color: #111111 !important;
        border-color: #111111 !important;
        box-shadow: none !important;
    }


    /* --------------------------------------------------------
       SUCHFELD
       -------------------------------------------------------- */

    .stTextInput input {
        border: 3px solid #111111 !important;
        border-radius: 2px !important;

        background: white !important;
        color: #111111 !important;

        font-size: 15px !important;
        font-weight: 700 !important;
    }

    .stTextInput label {
        color: #111111 !important;
        font-weight: 800 !important;
    }


    /* --------------------------------------------------------
       TEXTAREA
       -------------------------------------------------------- */

    .stTextArea textarea {
        border: 3px solid #111111 !important;
        border-radius: 2px !important;

        color: #111111 !important;
    }


    /* --------------------------------------------------------
       BILDER
       -------------------------------------------------------- */

    [data-testid="stImage"] img {
        border-radius: 0 !important;
    }


    /* --------------------------------------------------------
       FOOTER-ABSTAND
       -------------------------------------------------------- */

    .footer-space {
        height: 5px;
    }


    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media (max-width: 600px) {

        .main .block-container {
            max-width: 100%;

            padding-left: 7px;
            padding-right: 7px;

            padding-top: 5px;
            padding-bottom: 5px;
        }

        .stButton > button {
            min-height: 49px !important;
            font-size: 13px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

if "search_term" not in st.session_state:
    st.session_state.search_term = ""


# ============================================================
# NAVIGATION
# ============================================================

def go_to(page):

    st.session_state.page = page
    st.rerun()


# ============================================================
# LABEL BEREINIGEN
# ============================================================

def clean_label(label):

    if label is None:
        return ""

    label = str(label).strip()

    # Entfernt z.B.
    #
    # 0 Flasche
    # 1 Flasche
    # 2 - Tasche
    # 3. Rucksack
    # 4) Kleidung
    #
    # Ergebnis:
    #
    # Flasche
    # Tasche
    # Rucksack
    # Kleidung

    label = re.sub(
        r"^\s*\d+\s*[\.\)\-:\s]+\s*",
        "",
        label
    )

    return label.strip()


# ============================================================
# LABELS LADEN
# ============================================================

def load_labels():

    default_labels = [
        "Klasse 1",
        "Klasse 2",
        "Klasse 3",
        "Klasse 4",
        "Klasse 5",
    ]

    if not LABELS_PATH.exists():
        return default_labels

    labels = []

    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                line = clean_label(line)

                if line:
                    labels.append(line)

    except Exception:

        return default_labels

    if not labels:
        return default_labels

    return labels


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

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


# ============================================================
# FUNDSTÜCK SPEICHERN
# ============================================================

def add_item(
    image_path,
    name,
    kategorie,
    farbe,
    gefunden_am,
    fundort,
    notizen,
    aktueller_standort
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
            clean_label(name),
            clean_label(kategorie),
            farbe,
            gefunden_am,
            fundort,
            notizen,
            aktueller_standort,
            datetime.now().isoformat()
        )
    )

    connection.commit()

    item_id = cursor.lastrowid

    connection.close()

    return item_id


# ============================================================
# ALLE FUNDSTÜCKE
# ============================================================

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


# ============================================================
# EIN FUNDSTÜCK
# ============================================================

def get_item(item_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM fundstuecke
        WHERE id = ?
        """,
        (item_id,)
    )

    item = cursor.fetchone()

    connection.close()

    return item


# ============================================================
# ZUORDNUNG
# ============================================================

def update_assigned(item_id):

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE fundstuecke
        SET zugeordnet = 1
        WHERE id = ?
        """,
        (item_id,)
    )

    connection.commit()
    connection.close()


# ============================================================
# BILD LADEN
# ============================================================

def get_image(path):

    try:

        if path and Path(path).exists():

            return Image.open(path)

    except Exception:

        pass

    return None


# ============================================================
# BILD SPEICHERN
# ============================================================

def save_uploaded_image(uploaded_file):

    if uploaded_file is None:
        return None

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        filename = (
            f"{uuid.uuid4().hex}.jpg"
        )

        path = IMAGE_DIR / filename

        image.save(
            path,
            "JPEG",
            quality=92
        )

        return str(path)

    except Exception as error:

        st.error(
            f"Bild konnte nicht gespeichert werden: {error}"
        )

        return None


# ============================================================
# H5-MODELL REPARIEREN
# ============================================================

def repair_h5_model():

    if not MODEL_PATH.exists():

        return None


    if FIXED_MODEL_PATH.exists():

        try:

            if (
                FIXED_MODEL_PATH.stat().st_mtime
                >= MODEL_PATH.stat().st_mtime
            ):

                return FIXED_MODEL_PATH

        except Exception:

            pass


    try:

        import h5py

        shutil.copy2(
            MODEL_PATH,
            FIXED_MODEL_PATH
        )


        with h5py.File(
            FIXED_MODEL_PATH,
            "r+"
        ) as file:

            if "model_config" not in file.attrs:

                return FIXED_MODEL_PATH


            config = file.attrs[
                "model_config"
            ]


            if isinstance(
                config,
                bytes
            ):

                config = config.decode(
                    "utf-8"
                )


            config = json.loads(
                config
            )


            def clean_config(obj):

                if isinstance(
                    obj,
                    dict
                ):

                    if (
                        obj.get("class_name")
                        == "DepthwiseConv2D"
                    ):

                        obj.get(
                            "config",
                            {}
                        ).pop(
                            "groups",
                            None
                        )


                    for value in obj.values():

                        clean_config(value)


                elif isinstance(
                    obj,
                    list
                ):

                    for value in obj:

                        clean_config(value)


            clean_config(config)


            file.attrs[
                "model_config"
            ] = json.dumps(config)


        return FIXED_MODEL_PATH


    except Exception:

        return MODEL_PATH


# ============================================================
# KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_ai_model():

    model_file = repair_h5_model()


    if model_file is None:

        st.error(
            "Die Datei keras_model.h5 wurde nicht gefunden."
        )

        return None


    try:

        model = tf.keras.models.load_model(
            model_file,
            compile=False
        )

        return model


    except Exception as error:

        st.error(
            "Das KI-Modell konnte nicht geladen werden."
        )

        with st.expander(
            "Technische Fehlermeldung"
        ):

            st.code(
                str(error)
            )

        return None


# ============================================================
# BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_data = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    image_data = (
        image_data.astype(
            "float32"
        )
    )

    image_data = (
        image_data / 127.5
    ) - 1.0

    return tf.expand_dims(
        image_data,
        axis=0
    )


# ============================================================
# MODEL-LAYER AUSFÜHREN
# ============================================================

def run_layers(
    model_part,
    tensor
):

    if model_part is None:

        return tensor


    for layer in getattr(
        model_part,
        "layers",
        []
    ):

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):

            continue


        tensor = layer(
            tensor,
            training=False
        )


    return tensor


# ============================================================
# KI-ERKENNUNG
# ============================================================

def predict_image(image):

    model = load_ai_model()


    if model is None:

        return None


    try:

        tensor = prepare_image(
            image
        )


        outer_layers = list(
            getattr(
                model,
                "layers",
                []
            )
        )


        feature_model = None
        classifier_model = None


        # ----------------------------------------------------
        # Teachable-Machine-Struktur
        # ----------------------------------------------------

        for layer in outer_layers:

            name = getattr(
                layer,
                "name",
                ""
            )


            if name == "sequential_1":

                feature_model = layer


            elif name == "sequential_3":

                classifier_model = layer


        # ----------------------------------------------------
        # Falls die Namen anders sind
        # ----------------------------------------------------

        if (
            feature_model is None
            or classifier_model is None
        ):

            nested = []


            for layer in outer_layers:

                if hasattr(
                    layer,
                    "layers"
                ):

                    inner_layers = getattr(
                        layer,
                        "layers",
                        []
                    )


                    if len(
                        inner_layers
                    ) >= 2:

                        nested.append(
                            layer
                        )


            if len(nested) >= 2:

                feature_model = nested[0]
                classifier_model = nested[-1]


        # ----------------------------------------------------
        # Feature-Modell
        # ----------------------------------------------------

        if feature_model is not None:

            tensor = run_layers(
                feature_model,
                tensor
            )


            if classifier_model is not None:

                tensor = run_layers(
                    classifier_model,
                    tensor
                )


        else:

            tensor = run_layers(
                model,
                tensor
            )


        # ----------------------------------------------------
        # Wahrscheinlichkeiten
        # ----------------------------------------------------

        probabilities = (
            tensor.numpy()
            .reshape(-1)
        )


        total = float(
            probabilities.sum()
        )


        if (
            total < 0.99
            or total > 1.01
            or any(
                probabilities < 0
            )
        ):

            probabilities = (
                tf.nn.softmax(
                    probabilities
                ).numpy()
            )


        ranking = sorted(
            enumerate(
                probabilities
            ),
            key=lambda x: x[1],
            reverse=True
        )


        results = []


        for index, probability in ranking:

            if index < len(LABELS):

                label = LABELS[index]

            else:

                label = (
                    f"Klasse {index + 1}"
                )


            # Nummer vor dem Begriff entfernen
            label = clean_label(
                label
            )


            results.append(
                {
                    "label": label,
                    "confidence": float(
                        probability
                    )
                }
            )


        return results


    except Exception as error:

        st.error(
            "KI-Erkennung fehlgeschlagen."
        )


        with st.expander(
            "Technische Fehlermeldung"
        ):

            st.code(
                str(error)
            )


        return None


# ============================================================
# HEADER
# ============================================================

def show_header(
    show_back=False
):

    left, center, right = st.columns(
        [1, 5, 1]
    )


    # --------------------------------------------------------
    # ZURÜCK
    # --------------------------------------------------------

    with left:

        if show_back:

            if BACK_ICON_PATH.exists():

                st.image(
                    BACK_ICON_PATH,
                    width=24
                )


            if st.button(
                "←",
                key=f"back_{st.session_state.page}",
                width="content"
            ):

                go_to("start")


    # --------------------------------------------------------
    # FUNDBÜRO LOGO
    # --------------------------------------------------------

    with center:

        if LOGO_PATH.exists():

            st.image(
                LOGO_PATH,
                width=170
            )

        else:

            st.markdown(
                "## FUNDBÜRO"
            )


    # --------------------------------------------------------
    # PROFIL
    # --------------------------------------------------------

    with right:

        if PROFILE_ICON_PATH.exists():

            st.image(
                PROFILE_ICON_PATH,
                width=28
            )

        else:

            st.write("●")


# ============================================================
# NEUER FOOTER
# ============================================================

def show_footer():

    # kleiner Abstand zum letzten Inhalt
    st.write("")

    # ========================================================
    # DAS KOMPLETTE LOGO WIRD ALS EIN BILD ANGEZEIGT
    #
    # TU ES | KATHARINEUM ZU LÜBECK | STEUERRAD
    #
    # Dadurch bleibt die Anordnung exakt wie in deinem Bild.
    # ========================================================

    if FOOTER_LOGO_PATH.exists():

        st.image(
            FOOTER_LOGO_PATH,
            width=390
        )

    else:

        st.warning(
            "footer_logo.png wurde nicht gefunden."
        )


# ============================================================
# STARTSEITE
# ============================================================

def page_start():

    show_header()

    st.write("")


    # ========================================================
    # FOTO HOCHLADEN
    # ========================================================

    upload_columns = st.columns(
        [1, 8],
        gap="small"
    )


    with upload_columns[0]:

        if UPLOAD_ICON_PATH.exists():

            st.image(
                UPLOAD_ICON_PATH,
                width=25
            )


    with upload_columns[1]:

        upload_clicked = st.button(
            "FOTO HOCHLADEN",
            key="start_upload",
            width="stretch"
        )


    if upload_clicked:

        go_to("upload")


    st.write("")


    # ========================================================
    # KLEIDUNGSSTÜCK SUCHEN
    # ========================================================

    search_columns = st.columns(
        [1, 8],
        gap="small"
    )


    with search_columns[0]:

        if SEARCH_ICON_PATH.exists():

            st.image(
                SEARCH_ICON_PATH,
                width=25
            )


    with search_columns[1]:

        search_clicked = st.button(
            "KLEIDUNGSSTÜCK SUCHEN",
            key="start_search",
            width="stretch"
        )


    if search_clicked:

        go_to("search")


    st.write("")


    # ========================================================
    # LETZTE FUNDSTÜCKE
    # ========================================================

    st.subheader(
        "LETZTE FUNDSTÜCKE"
    )


    items = get_all_items()

    recent_items = items[:6]


    if not recent_items:

        st.info(
            "Noch keine Fundstücke vorhanden."
        )


    else:

        columns = st.columns(
            3,
            gap="small"
        )


        for index, item in enumerate(
            recent_items
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
                        width="stretch"
                    )


                label = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )


                label = clean_label(
                    label
                )


                if st.button(
                    label,
                    key=f"recent_{item['id']}",
                    width="stretch"
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go_to("detail")


    show_footer()


# ============================================================
# UPLOADSEITE
# ============================================================

def page_upload():

    show_header(
        show_back=True
    )


    st.subheader(
        "FUNDSTÜCK AUFNEHMEN"
    )


    # ========================================================
    # KAMERA
    # ========================================================

    camera_image = st.camera_input(
        "Foto aufnehmen"
    )


    # ========================================================
    # DATEI HOCHLADEN
    # ========================================================

    uploaded_file = st.file_uploader(
        "Oder Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    if camera_image is not None:

        selected_image = camera_image

    else:

        selected_image = uploaded_file


    if selected_image is not None:

        image = Image.open(
            selected_image
        ).convert("RGB")


        st.image(
            image,
            width="stretch"
        )


        # ====================================================
        # KI
        # ====================================================

        st.subheader(
            "KI-ERKENNUNG"
        )


        with st.spinner(
            "Fundstück wird erkannt..."
        ):

            results = predict_image(
                image
            )


        detected_name = ""


        if results:

            detected_name = clean_label(
                results[0]["label"]
            )


            confidence = (
                results[0]["confidence"]
                * 100
            )


            st.write(
                f"**Erkannt:** {detected_name}"
            )


            st.write(
                f"**Sicherheit:** {confidence:.1f} %"
            )


            with st.expander(
                "Weitere Ergebnisse"
            ):

                for result in results:

                    result_name = clean_label(
                        result["label"]
                    )


                    result_confidence = (
                        result["confidence"]
                        * 100
                    )


                    st.write(
                        f"{result_name}: "
                        f"{result_confidence:.1f} %"
                    )


        # ====================================================
        # INFORMATIONEN
        # ====================================================

        st.subheader(
            "INFORMATIONEN"
        )


        name = st.text_input(
            "Bezeichnung",
            value=detected_name
        )


        category = st.selectbox(
            "Kategorie",
            [
                "",
                "Kleidung",
                "Flasche",
                "Tasche",
                "Schulsachen",
                "Elektronik",
                "Sonstiges"
            ]
        )


        color = st.text_input(
            "Farbe",
            placeholder="z. B. schwarz"
        )


        found_date = st.date_input(
            "Gefunden am"
        )


        location = st.text_input(
            "Fundort",
            placeholder="z. B. Obere Turnhalle"
        )


        notes = st.text_area(
            "Notizen",
            placeholder="Weitere Informationen..."
        )


        current_location = st.text_input(
            "Aktueller Standort",
            value="Fundkiste"
        )


        st.write("")


        if st.button(
            "FUNDSTÜCK SPEICHERN",
            key="save_item",
            width="stretch"
        ):

            image_path = save_uploaded_image(
                selected_image
            )


            if image_path:

                item_id = add_item(
                    image_path=image_path,
                    name=name,
                    kategorie=category,
                    farbe=color,
                    gefunden_am=found_date.isoformat(),
                    fundort=location,
                    notizen=notes,
                    aktueller_standort=current_location
                )


                st.session_state.selected_item = (
                    item_id
                )


                go_to("detail")


    show_footer()


# ============================================================
# SUCHSEITE
# ============================================================

def page_search():

    show_header()


    # ========================================================
    # SUCHFELD
    # ========================================================

    search_term = st.text_input(
        "Suche",
        value=st.session_state.search_term,
        placeholder="(z.B.:) Flaschen"
    )


    st.session_state.search_term = (
        search_term
    )


    # ========================================================
    # ÜBERSCHRIFT
    # ========================================================

    if search_term:

        st.subheader(
            f'ERGEBNISSE FÜR „{search_term.upper()}“'
        )

    else:

        st.subheader(
            "ERGEBNISSE"
        )


    # ========================================================
    # ZURÜCK
    # ========================================================

    if st.button(
        "←",
        key="search_back",
        width="content"
    ):

        go_to("start")


    # ========================================================
    # FILTER
    # ========================================================

    with st.expander(
        "FILTER"
    ):

        color_filter = st.text_input(
            "Farbe"
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
                "Sonstiges"
            ]
        )


    # ========================================================
    # FUNDSTÜCKE FILTERN
    # ========================================================

    items = get_all_items()

    filtered_items = []


    search_lower = (
        search_term
        .strip()
        .lower()
    )


    color_lower = (
        color_filter
        .strip()
        .lower()
    )


    for item in items:

        searchable = " ".join(
            [
                item["name"] or "",
                item["kategorie"] or "",
                item["farbe"] or "",
                item["fundort"] or "",
                item["notizen"] or ""
            ]
        ).lower()


        if search_lower:

            if search_lower not in searchable:

                continue


        if color_lower:

            item_color = (
                item["farbe"] or ""
            ).lower()


            if color_lower not in item_color:

                continue


        if category_filter:

            if (
                item["kategorie"]
                != category_filter
            ):

                continue


        filtered_items.append(
            item
        )


    # ========================================================
    # ERGEBNISSE
    # ========================================================

    if not filtered_items:

        st.info(
            "Keine passenden Fundstücke gefunden."
        )


    else:

        columns = st.columns(
            3,
            gap="small"
        )


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
                        width="stretch"
                    )


                label = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )


                label = clean_label(
                    label
                )


                if st.button(
                    label,
                    key=f"result_{item['id']}",
                    width="stretch"
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )


                    go_to("detail")


    show_footer()


# ============================================================
# DETAILSEITE
# ============================================================

def page_detail():

    show_header()


    item_id = (
        st.session_state.selected_item
    )


    if item_id is None:

        go_to("search")


    item = get_item(
        item_id
    )


    if item is None:

        st.error(
            "Fundstück wurde nicht gefunden."
        )

        return


    # ========================================================
    # SUCHFELD
    # ========================================================

    st.text_input(
        "Suche",
        value=st.session_state.search_term,
        placeholder="(z.B.:) Flaschen",
        key="detail_search"
    )


    # ========================================================
    # ÜBERSCHRIFT
    # ========================================================

    if st.session_state.search_term:

        st.subheader(
            f'ERGEBNISSE FÜR „{st.session_state.search_term.upper()}“'
        )

    else:

        st.subheader(
            "FUNDSTÜCK"
        )


    # ========================================================
    # ZURÜCK
    # ========================================================

    if st.button(
        "←",
        key="detail_back",
        width="content"
    ):

        go_to("search")


    # ========================================================
    # BILD
    # ========================================================

    image = get_image(
        item["image_path"]
    )


    if image:

        image_columns = st.columns(
            [1, 2, 1]
        )


        with image_columns[1]:

            st.image(
                image,
                width="stretch"
            )


    # ========================================================
    # NAME
    # ========================================================

    title = (
        item["name"]
        or item["kategorie"]
        or "Fundstück"
    )


    title = clean_label(
        title
    )


    st.subheader(
        title.upper()
    )


    # ========================================================
    # INFORMATIONEN
    # ========================================================

    st.markdown(
        "**GEFUNDEN AM:**"
    )

    st.write(
        item["gefunden_am"] or "-"
    )

    st.divider()


    st.markdown(
        "**FUNDORT:**"
    )

    st.write(
        item["fundort"] or "-"
    )

    st.divider()


    st.markdown(
        "**NOTIZEN:**"
    )

    st.write(
        item["notizen"] or "-"
    )

    st.divider()


    st.markdown(
        "**AKTUELLER STANDORT:**"
    )

    st.write(
        item["aktueller_standort"] or "-"
    )


    if item["farbe"]:

        st.divider()

        st.markdown(
            "**FARBE:**"
        )

        st.write(
            item["farbe"]
        )


    st.write("")


    # ========================================================
    # ZUORDNUNG
    # ========================================================

    if item["zugeordnet"]:

        st.success(
            "FUNDSTÜCK ZUGEORDNET"
        )


    else:

        if st.button(
            "FUNDSTÜCK ZUORDNEN",
            key=f"assign_{item['id']}",
            width="stretch"
        ):

            update_assigned(
                item["id"]
            )

            st.rerun()


    show_footer()


# ============================================================
# APP STARTEN
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
