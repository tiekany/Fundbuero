import os
import re
import json
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# ============================================================
# TENSORFLOW / KERAS
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
    page_title="KATH-LOST AND FOUND",
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


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GRUNDLAYOUT
       ===================================================== */

    .stApp {
        background: white;
    }

    .main .block-container {
        max-width: 470px;
        padding-top: 20px;
        padding-bottom: 30px;
        padding-left: 14px;
        padding-right: 14px;
    }

    /* =====================================================
       STREAMLIT ELEMENTE VERSTECKEN
       ===================================================== */

    header {
        visibility: hidden;
        height: 0;
    }

    footer {
        visibility: hidden;
        height: 0;
    }

    /* =====================================================
       APP-RAHMEN
       ===================================================== */

    .app-frame {
        border: 4px solid #222222;
        border-radius: 38px;
        padding: 20px 16px 16px 16px;
        background: #ffffff;
        min-height: 650px;
    }

    /* =====================================================
       LOGO
       ===================================================== */

    .logo-area {
        text-align: center;
        margin-bottom: 12px;
    }

    .logo-area img {
        max-width: 190px;
        height: auto;
    }

    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border: 4px solid #222222 !important;
        border-radius: 3px !important;
        background: white !important;
        color: #111111 !important;
        font-size: 16px !important;
        font-weight: 900 !important;
        min-height: 52px !important;
        width: 100% !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #f3f3f3 !important;
        border-color: #222222 !important;
        color: #111111 !important;
    }

    .stButton > button:focus {
        border-color: #222222 !important;
        box-shadow: none !important;
    }

    /* =====================================================
       SUCHFELD
       ===================================================== */

    .stTextInput input {
        border: 4px solid #222222 !important;
        border-radius: 2px !important;
        font-size: 17px !important;
        font-weight: 800 !important;
        color: #111111 !important;
        background: white !important;
    }

    .stTextInput label {
        font-weight: 900 !important;
        color: #111111 !important;
    }

    /* =====================================================
       ÜBERSCHRIFTEN
       ===================================================== */

    h1,
    h2,
    h3 {
        color: #111111 !important;
        font-weight: 900 !important;
    }

    h3 {
        font-size: 20px !important;
        margin-top: 8px !important;
        margin-bottom: 10px !important;
    }

    /* =====================================================
       BILDRASTER
       ===================================================== */

    .grid-image {
        border: 4px solid #222222;
        border-radius: 2px;
        overflow: hidden;
        background: white;
    }

    /* =====================================================
       DETAILTABELLE
       ===================================================== */

    .detail-box {
        border: 4px solid #222222;
        margin-top: 12px;
    }

    .detail-row {
        border-bottom: 3px solid #222222;
        padding: 7px 8px;
        font-size: 14px;
        line-height: 1.25;
    }

    .detail-row:last-child {
        border-bottom: none;
    }

    .detail-label {
        font-weight: 900;
    }

    /* =====================================================
       GRÜNER STATUS
       ===================================================== */

    .assigned-box {
        border: 4px solid #138a2e;
        color: #138a2e;
        font-weight: 900;
        text-align: center;
        padding: 8px;
        margin-top: 0;
        font-size: 17px;
    }

    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-area {
        margin-top: 18px;
        text-align: center;
    }

    .footer-tu {
        color: #d00000;
        font-size: 25px;
        font-weight: 900;
        line-height: 0.85;
    }

    .footer-school {
        font-size: 10px;
        font-weight: 900;
        line-height: 1.0;
    }

    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 600px) {

        .main .block-container {
            width: 100%;
            max-width: 100%;
            padding-left: 8px;
            padding-right: 8px;
        }

        .app-frame {
            border-radius: 32px;
            padding: 16px 12px 12px 12px;
        }

        .stButton > button {
            font-size: 14px !important;
        }

        .detail-row {
            font-size: 13px;
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
# LABELS BEREINIGEN
# ============================================================

def clean_label(label):
    """
    Entfernt Nummerierungen:

    0 Flasche
    1. Flasche
    2) Kleidung
    03 - Tasche

    wird zu:

    Flasche
    Kleidung
    Tasche
    """

    label = str(label).strip()

    label = re.sub(
        r"^\s*\d+\s*[\.\)\-:]*\s*",
        "",
        label
    )

    return label.strip()


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

    return labels if labels else default_labels


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
        )
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
        (item_id,)
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
        (item_id,)
    )

    connection.commit()
    connection.close()


# ============================================================
# BILDER
# ============================================================

def get_image(path):

    try:

        if path and Path(path).exists():
            return Image.open(path)

    except Exception:
        pass

    return None


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
# H5 REPARATUR
# ============================================================

def repair_h5_model():

    if not MODEL_PATH.exists():
        return None

    if FIXED_MODEL_PATH.exists():

        try:

            if (
                FIXED_MODEL_PATH.stat().st_mtime
                >=
                MODEL_PATH.stat().st_mtime
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
# KI-MODELL
# ============================================================

@st.cache_resource
def load_ai_model():

    model_file = repair_h5_model()

    if model_file is None:
        return None

    try:

        return tf.keras.models.load_model(
            model_file,
            compile=False
        )

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

                    if len(
                        getattr(
                            layer,
                            "layers",
                            []
                        )
                    ) >= 2:

                        nested.append(
                            layer
                        )

            if len(nested) >= 2:

                feature_model = nested[0]
                classifier_model = nested[-1]

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
                label = f"Klasse {index + 1}"

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

def show_logo():

    if LOGO_PATH.exists():

        st.image(
            LOGO_PATH,
            width=180
        )

    else:

        st.markdown(
            "## FUNDBÜRO"
        )


def show_header(
    back=False
):

    # obere Zeile
    left, center, right = st.columns(
        [1, 5, 1]
    )

    with left:

        if back:

            if st.button(
                "←",
                key="header_back",
                width="content"
            ):

                go_to("search")

    with center:

        show_logo()

    with right:

        if PROFILE_ICON_PATH.exists():

            st.image(
                PROFILE_ICON_PATH,
                width=28
            )

        else:

            st.write("●")


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    st.write("")

    # Footer nach der Skizze
    footer_left, footer_middle, footer_right = st.columns(
        [1, 2, 1]
    )

    with footer_left:

        st.markdown(
            """
            <div class="footer-tu">
            TU<br>ES
            </div>
            """,
            unsafe_allow_html=True
        )

    with footer_middle:

        st.markdown(
            """
            <div class="footer-school">
            KATHARINEUM<br>
            ZU LÜBECK
            </div>
            """,
            unsafe_allow_html=True
        )

    with footer_right:

        if FOOTER_LOGO_PATH.exists():

            st.image(
                FOOTER_LOGO_PATH,
                width=65
            )

    st.write("")


# ============================================================
# STARTSEITE
# ABB. 1
# ============================================================

def page_start():

    # äußerer App-Rahmen
    st.markdown(
        '<div class="app-frame">',
        unsafe_allow_html=True
    )

    show_header()

    st.write("")

    # --------------------------------------------------------
    # FOTO HOCHLADEN
    # --------------------------------------------------------

    if UPLOAD_ICON_PATH.exists():

        icon_col, button_col = st.columns(
            [1, 8]
        )

        with icon_col:

            st.image(
                UPLOAD_ICON_PATH,
                width=25
            )

        with button_col:

            upload_clicked = st.button(
                "FOTO HOCHLADEN",
                key="start_upload",
                width="stretch"
            )

    else:

        upload_clicked = st.button(
            "FOTO HOCHLADEN",
            key="start_upload",
            width="stretch"
        )

    if upload_clicked:
        go_to("upload")


    st.write("")


    # --------------------------------------------------------
    # KLEIDUNGSSTÜCK SUCHEN
    # --------------------------------------------------------

    if SEARCH_ICON_PATH.exists():

        icon_col, button_col = st.columns(
            [1, 8]
        )

        with icon_col:

            st.image(
                SEARCH_ICON_PATH,
                width=25
            )

        with button_col:

            search_clicked = st.button(
                "KLEIDUNGSSTÜCK SUCHEN",
                key="start_search",
                width="stretch"
            )

    else:

        search_clicked = st.button(
            "KLEIDUNGSSTÜCK SUCHEN",
            key="start_search",
            width="stretch"
        )

    if search_clicked:
        go_to("search")


    st.write("")


    # --------------------------------------------------------
    # LETZTE FUNDSTÜCKE
    # --------------------------------------------------------

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

        columns = st.columns(3)

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

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# UPLOAD
# ============================================================

def page_upload():

    st.markdown(
        '<div class="app-frame">',
        unsafe_allow_html=True
    )

    show_header(
        back=True
    )

    st.subheader(
        "FUNDSTÜCK AUFNEHMEN"
    )

    camera_image = st.camera_input(
        "Foto aufnehmen"
    )

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    selected_image = (
        camera_image
        if camera_image is not None
        else uploaded_file
    )


    if selected_image is not None:

        image = Image.open(
            selected_image
        ).convert("RGB")

        st.image(
            image,
            width="stretch"
        )


        # ----------------------------------------------------
        # KI
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Daten
        # ----------------------------------------------------

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
            "Farbe"
        )

        found_date = st.date_input(
            "Gefunden am"
        )

        location = st.text_input(
            "Fundort"
        )

        notes = st.text_area(
            "Notizen"
        )

        current_location = st.text_input(
            "Aktueller Standort",
            value="Fundkiste"
        )


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
                    image_path,
                    name,
                    category,
                    color,
                    found_date.isoformat(),
                    location,
                    notes,
                    current_location
                )

                st.session_state.selected_item = (
                    item_id
                )

                go_to("detail")


    show_footer()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SUCHSEITE
# ABB. 2
# ============================================================

def page_search():

    st.markdown(
        '<div class="app-frame">',
        unsafe_allow_html=True
    )

    show_header()

    # --------------------------------------------------------
    # Suchfeld
    # --------------------------------------------------------

    search_term = st.text_input(
        "Suche",
        value=st.session_state.search_term,
        placeholder="(z.B.:) Flaschen"
    )

    st.session_state.search_term = (
        search_term
    )


    # --------------------------------------------------------
    # Überschrift
    # --------------------------------------------------------

    if search_term:

        st.subheader(
            f'ERGEBNISSE FÜR „{search_term.upper()}“'
        )

    else:

        st.subheader(
            "ERGEBNISSE"
        )


    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Zurück
    # --------------------------------------------------------

    if BACK_ICON_PATH.exists():

        if st.button(
            "←",
            key="search_back",
            width="content"
        ):

            go_to("start")

    else:

        if st.button(
            "←",
            key="search_back",
            width="content"
        ):

            go_to("start")


    # --------------------------------------------------------
    # Ergebnisse
    # --------------------------------------------------------

    items = get_all_items()

    filtered = []

    query = (
        search_term
        .strip()
        .lower()
    )

    color_query = (
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


        if query and query not in searchable:
            continue


        if color_query:

            if color_query not in (
                item["farbe"] or ""
            ).lower():

                continue


        if category_filter:

            if (
                item["kategorie"]
                != category_filter
            ):

                continue


        filtered.append(
            item
        )


    # --------------------------------------------------------
    # Raster
    # --------------------------------------------------------

    if not filtered:

        st.info(
            "Keine passenden Fundstücke gefunden."
        )

    else:

        columns = st.columns(3)

        for index, item in enumerate(
            filtered
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

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DETAILSEITE
# ABB. 3
# ============================================================

def page_detail():

    st.markdown(
        '<div class="app-frame">',
        unsafe_allow_html=True
    )

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
            "Fundstück nicht gefunden."
        )

        return


    # --------------------------------------------------------
    # Suchfeld wie in Abb. 3
    # --------------------------------------------------------

    st.text_input(
        "Suche",
        value=st.session_state.search_term,
        placeholder="(z.B.:) Flaschen",
        key="detail_search"
    )


    # --------------------------------------------------------
    # Ergebnisüberschrift
    # --------------------------------------------------------

    st.subheader(
        f'ERGEBNISSE FÜR „{st.session_state.search_term.upper()}“'
        if st.session_state.search_term
        else "FUNDSTÜCK"
    )


    # --------------------------------------------------------
    # Zurück-Pfeil
    # --------------------------------------------------------

    if st.button(
        "←",
        key="detail_back",
        width="content"
    ):

        go_to("search")


    # --------------------------------------------------------
    # Großes Bild
    # --------------------------------------------------------

    image = get_image(
        item["image_path"]
    )

    if image:

        st.image(
            image,
            width=230
        )


    # --------------------------------------------------------
    # Titel
    # --------------------------------------------------------

    title = (
        item["name"]
        or item["kategorie"]
        or "Fundstück"
    )

    title = clean_label(
        title
    )


    # --------------------------------------------------------
    # Detailinformationen
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="detail-box">

            <div class="detail-row">
                <span class="detail-label">
                    GEFUNDEN AM:
                </span>
                {item["gefunden_am"] or "-"}
            </div>

            <div class="detail-row">
                <span class="detail-label">
                    FUNDORT:
                </span>
                {item["fundort"] or "-"}
            </div>

            <div class="detail-row">
                <span class="detail-label">
                    NOTIZEN:
                </span>
                {item["notizen"] or "-"}
            </div>

            <div class="detail-row">
                <span class="detail-label">
                    AKTUELLER STANDORT:
                </span>
                {item["aktueller_standort"] or "-"}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Zuordnung
    # --------------------------------------------------------

    if item["zugeordnet"]:

        st.markdown(
            """
            <div class="assigned-box">
                FUNDSTÜCK ZUGEORDNET
            </div>
            """,
            unsafe_allow_html=True
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

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# APP-ROUTER
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
