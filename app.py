import os

# ============================================================
# TENSORFLOW / KERAS
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import sqlite3
import uuid
import hashlib

from pathlib import Path
from datetime import datetime

import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# SEITENEINSTELLUNGEN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "modell"
IMAGE_DIR = BASE_DIR / "bilder"
DATA_DIR = BASE_DIR / "daten"

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"
DB_PATH = DATA_DIR / "fundbuero.db"

LOGO_PATH = ASSETS_DIR / "katharineum_logo.png"


# Ordner automatisch erstellen
ASSETS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "⌕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GRUNDLEGENDE DESIGNWERTE
       ======================================================== */

    :root {
        --black: #000000;
        --white: #ffffff;
        --background: #f3f3f3;

        --border: 2px solid #000000;
        --radius: 8px;
    }


    /* ========================================================
       GESAMTE APP
       ======================================================== */

    .stApp {
        background: var(--background);
    }


    /* ========================================================
       GESAMTER BILDSCHIRM
       ======================================================== */

    .block-container {
        width: 100%;
        max-width: 1200px;

        margin: 0 auto;

        padding-top: 20px;
        padding-left: 24px;
        padding-right: 24px;
        padding-bottom: 120px;
    }


    * {
        box-sizing: border-box;

        font-family:
            Arial,
            Helvetica,
            sans-serif;
    }


    /* ========================================================
       STREAMLIT STANDARD-BEREICHE
       ======================================================== */

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }


    /* ========================================================
       HAUPTTITEL
       ======================================================== */

    .app-title {
        width: 100%;

        text-align: center;

        font-size: 36px;
        font-weight: 900;

        letter-spacing: 1px;

        color: #000000;

        margin-top: 4px;
        margin-bottom: 20px;
    }


    /* ========================================================
       SCHULLOGO
       ======================================================== */

    .logo-container {
        width: 100%;

        display: flex;
        justify-content: center;
        align-items: center;

        margin-bottom: 12px;
    }

    .school-logo {
        max-width: 220px;
        max-height: 100px;

        object-fit: contain;
    }


    /* ========================================================
       ABSCHNITTSÜBERSCHRIFTEN
       ======================================================== */

    .section-title {
        color: #000000;

        font-size: 22px;
        font-weight: 800;

        margin-top: 26px;
        margin-bottom: 12px;
    }


    /* ========================================================
       ALLE BUTTONS
       ======================================================== */

    div.stButton > button {

        width: 100%;

        min-height: 54px;

        background: #ffffff;
        color: #000000;

        border: var(--border);
        border-radius: var(--radius);

        font-size: 16px;
        font-weight: 700;

        box-shadow: none;

        transition:
            background 0.15s ease,
            transform 0.05s ease;
    }


    div.stButton > button:hover {

        background: #eeeeee;

        color: #000000;

        border: var(--border);

        box-shadow: none;
    }


    div.stButton > button:focus {

        background: #ffffff;

        color: #000000;

        border: var(--border);

        box-shadow: none;
    }


    div.stButton > button:active {

        background: #eeeeee;

        color: #000000;

        border: var(--border);

        box-shadow: none;
    }


    /* ========================================================
       TEXT INPUT
       ======================================================== */

    div[data-baseweb="input"] {

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);

        box-shadow: none;
    }


    div[data-baseweb="input"]:focus-within {

        border: var(--border);

        box-shadow: none;
    }


    div[data-baseweb="input"] input {

        color: #000000;

        font-size: 16px;
    }


    /* ========================================================
       TEXTAREA
       ======================================================== */

    div[data-baseweb="textarea"] {

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);

        box-shadow: none;
    }


    div[data-baseweb="textarea"]:focus-within {

        border: var(--border);

        box-shadow: none;
    }


    div[data-baseweb="textarea"] textarea {

        color: #000000;

        font-size: 16px;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] > div {

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);

        box-shadow: none;
    }


    /* ========================================================
       DATE INPUT
       ======================================================== */

    div[data-testid="stDateInput"] div[data-baseweb="input"] {

        border: var(--border);
        border-radius: var(--radius);

        background: #ffffff;
    }


    /* ========================================================
       DATEI-UPLOAD
       ======================================================== */

    [data-testid="stFileUploader"] {

        width: 100%;

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);

        padding: 8px;

        box-shadow: none;
    }


    [data-testid="stFileUploaderDropzone"] {

        background: #ffffff;

        border: none;
    }


    /* ========================================================
       KAMERA
       ======================================================== */

    [data-testid="stCameraInput"] {

        width: 100%;

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);

        overflow: hidden;
    }


    /* ========================================================
       BILDER
       ======================================================== */

    [data-testid="stImage"] {

        width: 100%;
    }


    [data-testid="stImage"] img {

        border: var(--border);
        border-radius: var(--radius);

        background: #ffffff;

        width: 100%;
    }


    /* ========================================================
       FUNDSTÜCK-KARTEN
       ======================================================== */

    .item-card {

        width: 100%;

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);

        overflow: hidden;

        margin-bottom: 10px;
    }


    .item-card-image {

        width: 100%;

        aspect-ratio: 1 / 1;

        object-fit: cover;

        display: block;
    }


    /* ========================================================
       KI-ERGEBNIS
       ======================================================== */

    .ai-box {

        width: 100%;

        background: #ffffff;
        color: #000000;

        border: var(--border);
        border-radius: var(--radius);

        padding: 18px;

        margin-top: 16px;
        margin-bottom: 16px;

        text-align: center;

        font-size: 19px;
        font-weight: 700;
    }


    /* ========================================================
       INFORMATIONSKÄSTEN
       ======================================================== */

    .info-box {

        width: 100%;

        background: #ffffff;
        color: #000000;

        border: var(--border);
        border-radius: var(--radius);

        padding: 16px;

        margin-top: 10px;
        margin-bottom: 10px;

        font-size: 16px;

        line-height: 1.5;
    }


    /* ========================================================
       ERFOLGSBOX
       ======================================================== */

    .success-box {

        width: 100%;

        background: #ffffff;
        color: #000000;

        border: var(--border);
        border-radius: var(--radius);

        padding: 16px;

        margin-top: 16px;

        text-align: center;

        font-size: 16px;
        font-weight: 800;
    }


    /* ========================================================
       DETAILBILD
       ======================================================== */

    .detail-image {

        width: 100%;

        max-height: 600px;

        object-fit: contain;

        background: #ffffff;

        border: var(--border);
        border-radius: var(--radius);
    }


    /* ========================================================
       TRENNLINIEN
       ======================================================== */

    hr {

        border: none;

        border-top: var(--border);

        margin-top: 24px;
        margin-bottom: 24px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {

        width: 100%;

        text-align: center;

        margin-top: 40px;

        padding-top: 22px;
        padding-bottom: 20px;

        border-top: var(--border);
    }


    .footer-tu-es {

        font-size: 28px;

        font-weight: 900;

        letter-spacing: 1px;
    }


    .footer-school {

        font-size: 15px;

        font-weight: 700;

        margin-top: 3px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .block-container {

            padding-left: 12px;
            padding-right: 12px;

            padding-top: 12px;
        }


        .app-title {

            font-size: 29px;
        }


        .section-title {

            font-size: 19px;
        }


        div.stButton > button {

            min-height: 50px;

            font-size: 15px;
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


if "search" not in st.session_state:
    st.session_state.search = ""


if "ai_result" not in st.session_state:
    st.session_state.ai_result = ""


if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = 0.0


if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None


# ============================================================
# NAVIGATION
# ============================================================

def go(page):

    st.session_state.page = page

    st.rerun()


# ============================================================
# DATENBANK
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH
    )


def init_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fundstuecke (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image_path TEXT,

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

    conn.commit()

    conn.close()


init_database()


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
    aktueller_standort,
):

    conn = get_connection()

    cursor = conn.cursor()

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

    conn.commit()

    conn.close()


# ============================================================
# FUNDSTÜCKE LADEN
# ============================================================

def get_latest_items(limit=8):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *

        FROM fundstuecke

        ORDER BY id DESC

        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_all_items():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *

        FROM fundstuecke

        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_item(item_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *

        FROM fundstuecke

        WHERE id = ?
        """,
        (item_id,),
    )

    row = cursor.fetchone()

    conn.close()

    return row


# ============================================================
# FUNDSTÜCK ZUORDNEN
# ============================================================

def mark_assigned(item_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE fundstuecke

        SET zugeordnet = 1

        WHERE id = ?
        """,
        (item_id,),
    )

    conn.commit()

    conn.close()


# ============================================================
# BILDER SPEICHERN
# ============================================================

def save_uploaded_image(uploaded_file):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    ]:

        suffix = ".jpg"


    filename = (
        uuid.uuid4().hex
        + suffix
    )


    path = IMAGE_DIR / filename


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    image.save(
        path,
        quality=90,
    )


    return str(path)


# ============================================================
# BILD LADEN
# ============================================================

def load_image(path):

    try:

        if path and Path(path).exists():

            return Image.open(path)

    except Exception:

        return None

    return None


# ============================================================
# LABELS
# ============================================================

def clean_label(label):

    if not label:

        return ""


    label = str(label).strip()


    label = re.sub(
        r"^\d+\s*",
        "",
        label,
    )


    label = re.sub(
        r"^\d+[\.\)\-:]\s*",
        "",
        label,
    )


    return label.strip()


def load_labels():

    if LABELS_PATH.exists():

        try:

            lines = LABELS_PATH.read_text(
                encoding="utf-8"
            ).splitlines()


            labels = [

                clean_label(line)

                for line in lines

                if line.strip()
            ]


            if labels:

                return labels


        except Exception:

            pass


    return [
        "Flasche",
        "Tasche",
        "Rucksack",
        "Kleidungsstück",
        "Sonstiges",
    ]


LABELS = load_labels()


# ============================================================
# KI-MODELL LADEN
# ============================================================

@st.cache_resource
def load_fast_model():

    if not MODEL_PATH.exists():

        return None


    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )


        # Modell einmal aufwärmen

        dummy = tf.zeros(
            (
                1,
                224,
                224,
                3,
            ),
            dtype=tf.float32,
        )


        try:

            model(
                dummy,
                training=False,
            )

        except Exception:

            pass


        return model


    except Exception:

        return None


# ============================================================
# KI PREDICTION FUNCTION
# ============================================================

@st.cache_resource
def get_predict_function():

    model = load_fast_model()


    if model is None:

        return None


    @tf.function(
        reduce_retracing=True
    )
    def predict_function(x):

        return model(
            x,
            training=False,
        )


    return predict_function


# ============================================================
# BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")


    image = image.resize(
        (
            224,
            224,
        ),
        Image.Resampling.BILINEAR,
    )


    tensor = tf.convert_to_tensor(
        image,
        dtype=tf.float32,
    )


    tensor = (
        tensor / 127.5
    ) - 1.0


    tensor = tf.expand_dims(
        tensor,
        axis=0,
    )


    return tensor


# ============================================================
# KI ERKENNUNG
# ============================================================

def predict_fast(image):

    predict_function = (
        get_predict_function()
    )


    if predict_function is None:

        return "", 0.0


    try:

        tensor = prepare_image(
            image
        )


        prediction = predict_function(
            tensor
        )


        prediction = (
            prediction.numpy()[0]
        )


        index = int(
            prediction.argmax()
        )


        confidence = float(
            prediction[index]
        )


        if index < len(LABELS):

            label = LABELS[index]

        else:

            label = (
                f"Klasse {index + 1}"
            )


        return (
            clean_label(label),
            confidence,
        )


    except Exception:

        return "", 0.0


# ============================================================
# HEADER
# ============================================================

def show_header(back=False):

    # --------------------------------------------------------
    # OHNE ZURÜCK
    # --------------------------------------------------------

    if not back:

        if LOGO_PATH.exists():

            st.markdown(
                '<div class="logo-container">',
                unsafe_allow_html=True,
            )

            st.image(
                str(LOGO_PATH),
                width=220,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


        st.markdown(
            """
            <div class="app-title">
                FUNDBÜRO
            </div>
            """,
            unsafe_allow_html=True,
        )

        return


    # --------------------------------------------------------
    # MIT ZURÜCK UND PROFIL
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [1, 8, 1]
    )


    with col1:

        if st.button(
            "←",
            key="header_back",
        ):

            go("start")


    with col2:

        st.markdown(
            """
            <div class="app-title">
                FUNDBÜRO
            </div>
            """,
            unsafe_allow_html=True,
        )


    with col3:

        # EINHEITLICHES PROFIL-ICON

        if st.button(
            "♙",
            key="header_profile",
        ):

            go("profile")


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    st.markdown(
        "<hr>",
        unsafe_allow_html=True,
    )


    col1, col2, col3 = st.columns(
        3
    )


    # START

    with col1:

        if st.button(
            "⌂",
            key="nav_home",
        ):

            go("start")


    # SUCHE

    with col2:

        if st.button(
            "⌕",
            key="nav_search",
        ):

            go("search")


    # PROFIL

    with col3:

        if st.button(
            "♙",
            key="nav_profile",
        ):

            go("profile")


    st.markdown(
        """
        <div class="app-footer">

            <div class="footer-tu-es">
                TU ES
            </div>

            <div class="footer-school">
                KATHARINEUM ZU LÜBECK
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FUNDSTÜCK-GITTER
# ============================================================

def show_item_grid(items):

    if not items:

        st.info(
            "Noch keine Fundstücke vorhanden."
        )

        return


    # ========================================================
    # RESPONSIVES RASTER
    #
    # Auf großen Bildschirmen:
    # 4 Spalten
    #
    # Auf kleineren Geräten:
    # Streamlit passt die Breite automatisch an.
    # ========================================================

    columns = st.columns(
        4,
        gap="medium",
    )


    for index, item in enumerate(items):

        column = columns[
            index % 4
        ]


        with column:

            item_id = item[0]

            image_path = item[1]

            name = (
                item[2]
                or "Fundstück"
            )


            image = load_image(
                image_path
            )


            if image is not None:

                st.image(
                    image,
                    width="stretch",
                )

            else:

                st.markdown(
                    """
                    <div class="info-box">
                        Kein Bild
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            if st.button(
                name,
                key=f"item_{item_id}",
            ):

                st.session_state.selected_item = (
                    item_id
                )

                go("detail")


# ============================================================
# STARTSEITE
# ============================================================

def start_page():

    show_header(
        back=False
    )


    # ========================================================
    # HAUPTAKTIONEN
    # ========================================================

    col1, col2 = st.columns(
        2,
        gap="medium",
    )


    with col1:

        if st.button(
            "↑  FOTO HOCHLADEN",
            key="start_upload",
        ):

            go("upload")


    with col2:

        if st.button(
            "⌕  FUNDSTÜCK SUCHEN",
            key="start_search",
        ):

            go("search")


    # ========================================================
    # LETZTE FUNDSTÜCKE
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            LETZTE FUNDSTÜCKE
        </div>
        """,
        unsafe_allow_html=True,
    )


    items = get_latest_items(
        8
    )


    show_item_grid(
        items
    )


    show_footer()


# ============================================================
# UPLOAD-SEITE
# ============================================================

def upload_page():

    show_header(
        back=True
    )


    st.markdown(
        """
        <div class="section-title">
            FUNDSTÜCK HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # KAMERA
    # ========================================================

    camera = st.camera_input(
        "Foto aufnehmen",
        key="camera_input",
    )


    # ========================================================
    # DATEI
    # ========================================================

    uploaded = st.file_uploader(
        "Oder Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="file_uploader",
    )


    # Kamera hat Vorrang

    file = (
        camera
        if camera is not None
        else uploaded
    )


    if file is None:

        st.info(
            "Lade ein Foto hoch oder "
            "nimm eines mit der Kamera auf."
        )

        show_footer()

        return


    # ========================================================
    # BILD ÖFFNEN
    # ========================================================

    image_bytes = file.getvalue()


    # Hash verhindert,
    # dass die KI bei jedem Streamlit-Rerun
    # erneut gestartet wird.

    image_id = hashlib.md5(
        image_bytes
    ).hexdigest()


    try:

        image = Image.open(
            file
        ).convert("RGB")


    except Exception:

        st.error(
            "Das Bild konnte nicht geöffnet werden."
        )

        show_footer()

        return


    # ========================================================
    # BILD ANZEIGEN
    # ========================================================

    st.image(
        image,
        width="stretch",
    )


    # ========================================================
    # AUTOMATISCHE KI
    # ========================================================

    if (
        st.session_state.last_image_id
        != image_id
    ):

        st.session_state.last_image_id = (
            image_id
        )


        st.session_state.ai_result = ""

        st.session_state.ai_confidence = (
            0.0
        )


        with st.spinner(
            "Fundstück wird erkannt..."
        ):

            label, confidence = predict_fast(
                image
            )


        st.session_state.ai_result = (
            label
        )


        st.session_state.ai_confidence = (
            confidence
        )


    label = (
        st.session_state.ai_result
    )


    confidence = (
        st.session_state.ai_confidence
    )


    # ========================================================
    # KI-ERGEBNIS
    # ========================================================

    if label:

        st.markdown(
            f"""
            <div class="ai-box">

                Erkannt: {label}

                <br>

                Sicherheit:
                {confidence * 100:.0f} %

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="ai-box">

                Keine sichere Erkennung möglich

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # ANGABEN
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            ANGABEN ZUM FUNDSTÜCK
        </div>
        """,
        unsafe_allow_html=True,
    )


    default_name = (
        label
        if label
        else ""
    )


    name = st.text_input(
        "Bezeichnung",
        value=default_name,
        key=f"name_{image_id}",
    )


    kategorie = st.text_input(
        "Kategorie",
        value=label,
        key=f"kategorie_{image_id}",
    )


    farbe = st.text_input(
        "Farbe",
        key=f"farbe_{image_id}",
    )


    gefunden_am = st.date_input(
        "Gefunden am",
        value=datetime.now().date(),
        key=f"datum_{image_id}",
    )


    fundort = st.text_input(
        "Fundort",
        key=f"fundort_{image_id}",
    )


    notizen = st.text_area(
        "Notizen",
        key=f"notizen_{image_id}",
    )


    aktueller_standort = st.text_input(
        "Aktueller Standort",
        value="Fundkiste",
        key=f"standort_{image_id}",
    )


    # ========================================================
    # SPEICHERN
    # ========================================================

    if st.button(
        "FUNDSTÜCK SPEICHERN",
        key=f"save_{image_id}",
    ):

        if not name.strip():

            st.error(
                "Bitte eine Bezeichnung eingeben."
            )

        else:

            saved_path = (
                save_uploaded_image(
                    file
                )
            )


            add_item(
                image_path=saved_path,
                name=name.strip(),
                kategorie=kategorie.strip(),
                farbe=farbe.strip(),
                gefunden_am=str(
                    gefunden_am
                ),
                fundort=fundort.strip(),
                notizen=notizen.strip(),
                aktueller_standort=(
                    aktueller_standort.strip()
                ),
            )


            # KI zurücksetzen

            st.session_state.ai_result = ""

            st.session_state.ai_confidence = (
                0.0
            )

            st.session_state.last_image_id = (
                None
            )


            st.success(
                "Fundstück wurde gespeichert."
            )


            go("start")


    show_footer()


# ============================================================
# SUCHSEITE
# ============================================================

def search_page():

    show_header(
        back=True
    )


    st.markdown(
        """
        <div class="section-title">
            FUNDSTÜCK SUCHEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # SUCHFELD
    # ========================================================

    search = st.text_input(
        "⌕",
        placeholder="z. B. Flasche",
        value=st.session_state.search,
        key="search_input",
    )


    st.session_state.search = search


    items = get_all_items()


    # ========================================================
    # FILTER
    # ========================================================

    if search.strip():

        query = search.strip().lower()


        filtered = []


        for item in items:

            searchable_text = " ".join(
                [
                    str(item[2] or ""),
                    str(item[3] or ""),
                    str(item[4] or ""),
                    str(item[6] or ""),
                    str(item[7] or ""),
                    str(item[8] or ""),
                ]
            ).lower()


            if query in searchable_text:

                filtered.append(
                    item
                )


        items = filtered


        st.markdown(
            f"""
            <div class="section-title">
                ERGEBNISSE FÜR
                „{search.upper()}“
            </div>
            """,
            unsafe_allow_html=True,
        )


    else:

        st.markdown(
            """
            <div class="section-title">
                ALLE FUNDSTÜCKE
            </div>
            """,
            unsafe_allow_html=True,
        )


    show_item_grid(
        items
    )


    show_footer()


# ============================================================
# DETAILSEITE
# ============================================================

def detail_page():

    item_id = (
        st.session_state.selected_item
    )


    if item_id is None:

        go("start")

        return


    item = get_item(
        item_id
    )


    if item is None:

        go("start")

        return


    show_header(
        back=True
    )


    # ========================================================
    # DATEN
    # ========================================================

    image_path = item[1]

    name = item[2]

    kategorie = item[3]

    farbe = item[4]

    gefunden_am = item[5]

    fundort = item[6]

    notizen = item[7]

    aktueller_standort = item[8]

    zugeordnet = item[9]


    # ========================================================
    # DETAILBILD
    # ========================================================

    image = load_image(
        image_path
    )


    if image is not None:

        st.image(
            image,
            width="stretch",
        )


    # ========================================================
    # TITEL
    # ========================================================

    st.markdown(
        f"""
        <div class="section-title">
            {name}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # INFORMATIONEN
    # ========================================================

    st.markdown(
        f"""
        <div class="info-box">

            <strong>GEFUNDEN AM:</strong>
            <br>

            {gefunden_am}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="info-box">

            <strong>FUNDORT:</strong>
            <br>

            {fundort or "Keine Angabe"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="info-box">

            <strong>KATEGORIE:</strong>
            <br>

            {kategorie or "Keine Angabe"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="info-box">

            <strong>FARBE:</strong>
            <br>

            {farbe or "Keine Angabe"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="info-box">

            <strong>NOTIZEN:</strong>
            <br>

            {notizen or "Keine Notizen"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="info-box">

            <strong>AKTUELLER STANDORT:</strong>
            <br>

            {
                aktueller_standort
                or "Keine Angabe"
            }

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # ZUORDNUNG
    # ========================================================

    if zugeordnet:

        st.markdown(
            """
            <div class="success-box">

                FUNDSTÜCK ZUGEORDNET

            </div>
            """,
            unsafe_allow_html=True,
        )


    else:

        if st.button(
            "FUNDSTÜCK ZUORDNEN",
            key=f"assign_{item_id}",
        ):

            mark_assigned(
                item_id
            )

            st.rerun()


    show_footer()


# ============================================================
# PROFILSEITE
# ============================================================

def profile_page():

    show_header(
        back=True
    )


    st.markdown(
        """
        <div class="section-title">
            PROFIL
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # LOGO
    # ========================================================

    if LOGO_PATH.exists():

        st.markdown(
            '<div class="logo-container">',
            unsafe_allow_html=True,
        )

        st.image(
            str(LOGO_PATH),
            width=220,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # SCHULE
    # ========================================================

    st.markdown(
        """
        <div class="info-box">

            <strong>
                KATHARINEUM ZU LÜBECK
            </strong>

            <br><br>

            Digitales Fundbüro für die Schule.

            <br><br>

            Fundstücke können fotografiert,
            automatisch erkannt,
            gespeichert und gesucht werden.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # FUNKTIONEN
    # ========================================================

    st.markdown(
        """
        <div class="info-box">

            <strong>FUNKTIONEN</strong>

            <br><br>

            • Fundstück fotografieren
            <br>

            • automatische KI-Erkennung
            <br>

            • Fundstücke speichern
            <br>

            • Fundstücke suchen
            <br>

            • Fundstücke zuordnen

        </div>
        """,
        unsafe_allow_html=True,
    )


    show_footer()


# ============================================================
# SEITEN-ROUTER
# ============================================================

if st.session_state.page == "start":

    start_page()


elif st.session_state.page == "upload":

    upload_page()


elif st.session_state.page == "search":

    search_page()


elif st.session_state.page == "detail":

    detail_page()


elif st.session_state.page == "profile":

    profile_page()


else:

    start_page()
    

