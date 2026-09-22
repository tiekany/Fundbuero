# ============================================================
# FUNDBÜRO - KATHARINEUM ZU LÜBECK
# Design bewusst nah an der ursprünglichen Skizze
# ============================================================

import os

# ------------------------------------------------------------
# KERAS / TENSORFLOW
# ------------------------------------------------------------

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
# PFADE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "modell"
IMAGE_DIR = BASE_DIR / "bilder"
DATA_DIR = BASE_DIR / "daten"

DB_PATH = DATA_DIR / "fundbuero.db"

LOGO_PATH = ASSETS_DIR / "katharineum_logo.png"
PROFILE_PATH = ASSETS_DIR / "profile.png"

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"


# ============================================================
# ORDNER ERSTELLEN
# ============================================================

ASSETS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GRUNDLAGE
       ====================================================== */

    :root {
        --black: #000000;
        --white: #ffffff;
        --light: #f4f4f4;

        --border: 2px solid #000000;

        --content-width: 980px;
    }


    .stApp {
        background: var(--light);
    }


    .block-container {

        max-width: var(--content-width);

        margin: 0 auto;

        padding-top: 18px;
        padding-left: 24px;
        padding-right: 24px;
        padding-bottom: 150px;
    }


    * {
        font-family:
            Arial,
            Helvetica,
            sans-serif;
    }


    /* ======================================================
       STREAMLIT AUFRÄUMEN
       ====================================================== */

    [data-testid="stHeader"] {
        background: transparent;
    }


    [data-testid="stToolbar"] {
        visibility: hidden;
    }


    /* ======================================================
       OBERER BEREICH
       ====================================================== */

    .top-header {

        position: relative;

        width: 100%;

        min-height: 75px;

        display: flex;

        justify-content: center;

        align-items: center;

        margin-bottom: 14px;
    }


    .main-title {

        margin: 0;

        color: #000000;

        font-size: 38px;

        line-height: 1;

        font-weight: 900;

        letter-spacing: 1px;

        text-align: center;
    }


    /* ======================================================
       PROFIL
       ====================================================== */

    .profile-area {

        position: absolute;

        right: 0;

        top: 0;

        width: 48px;

        height: 48px;

        display: flex;

        justify-content: center;

        align-items: center;
    }


    .profile-image {

        width: 42px;

        height: 42px;

        object-fit: contain;
    }


    /* ======================================================
       LOGO
       ====================================================== */

    .school-logo-area {

        display: flex;

        justify-content: center;

        align-items: center;

        margin-top: 3px;

        margin-bottom: 18px;
    }


    .school-logo {

        max-width: 210px;

        max-height: 90px;

        object-fit: contain;
    }


    /* ======================================================
       HAUPTAKTIONEN
       ====================================================== */

    .main-action {

        background: #ffffff;

        border: 3px solid #000000;

        border-radius: 4px;

        min-height: 92px;

        display: flex;

        align-items: center;

        justify-content: center;

        text-align: center;

        font-size: 20px;

        font-weight: 800;

        letter-spacing: 0.3px;

        margin-bottom: 14px;
    }


    /* ======================================================
       ABSCHNITTSÜBERSCHRIFTEN
       ====================================================== */

    .sketch-heading {

        color: #000000;

        font-size: 23px;

        font-weight: 900;

        margin-top: 27px;

        margin-bottom: 13px;

        letter-spacing: 0.3px;
    }


    /* ======================================================
       BILDER
       ====================================================== */

    .item-image {

        width: 100%;

        aspect-ratio: 1 / 1;

        object-fit: cover;

        background: #ffffff;

        border: 3px solid #000000;

        border-radius: 0;
    }


    .empty-image {

        width: 100%;

        aspect-ratio: 1 / 1;

        display: flex;

        align-items: center;

        justify-content: center;

        background: #ffffff;

        border: 3px solid #000000;

        font-size: 13px;

        font-weight: 700;
    }


    .item-caption {

        text-align: center;

        font-size: 14px;

        font-weight: 800;

        margin-top: 5px;

        margin-bottom: 15px;
    }


    /* ======================================================
       SUCHFELD
       ====================================================== */

    .search-label {

        font-size: 18px;

        font-weight: 800;

        margin-bottom: 7px;
    }


    div[data-baseweb="input"] {

        background: #ffffff;

        border: 3px solid #000000;

        border-radius: 3px;

        box-shadow: none;
    }


    div[data-baseweb="input"]:focus-within {

        border: 3px solid #000000;

        box-shadow: none;
    }


    div[data-baseweb="input"] input {

        color: #000000;

        font-size: 17px;
    }


    /* ======================================================
       TEXTFELDER
       ====================================================== */

    div[data-baseweb="textarea"] {

        background: #ffffff;

        border: 2px solid #000000;

        border-radius: 3px;

        box-shadow: none;
    }


    div[data-baseweb="textarea"]:focus-within {

        border: 2px solid #000000;

        box-shadow: none;
    }


    div[data-baseweb="textarea"] textarea {

        color: #000000;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    div.stButton > button {

        background: #ffffff;

        color: #000000;

        border: 2px solid #000000;

        border-radius: 4px;

        box-shadow: none;

        font-weight: 800;

        min-height: 46px;
    }


    div.stButton > button:hover {

        background: #000000;

        color: #ffffff;

        border: 2px solid #000000;
    }


    div.stButton > button:focus {

        background: #ffffff;

        color: #000000;

        border: 2px solid #000000;

        box-shadow: none;
    }


    /* ======================================================
       DETAIL-INFORMATIONEN
       ====================================================== */

    .detail-row {

        background: #ffffff;

        border: 2px solid #000000;

        border-radius: 0;

        padding: 13px 15px;

        margin-bottom: 9px;

        font-size: 15px;

        line-height: 1.45;
    }


    .detail-label {

        font-weight: 900;

        font-size: 13px;

        margin-bottom: 3px;
    }


    /* ======================================================
       KI ERGEBNIS
       ====================================================== */

    .ai-box {

        background: #ffffff;

        border: 3px solid #000000;

        border-radius: 3px;

        padding: 14px;

        margin-top: 12px;

        margin-bottom: 18px;

        text-align: center;

        font-size: 18px;

        font-weight: 900;
    }


    /* ======================================================
       ZUGEORDNET
       ====================================================== */

    .assigned {

        background: #ffffff;

        border: 3px solid #000000;

        border-radius: 3px;

        padding: 16px;

        text-align: center;

        font-size: 16px;

        font-weight: 900;

        margin-top: 14px;

        margin-bottom: 14px;
    }


    /* ======================================================
       TRENNER
       ====================================================== */

    .sketch-line {

        width: 100%;

        border-top: 2px solid #000000;

        margin-top: 28px;

        margin-bottom: 18px;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        text-align: center;

        margin-top: 35px;

        padding-top: 22px;

        padding-bottom: 10px;

        border-top: 3px solid #000000;
    }


    .footer-tu-es {

        font-size: 29px;

        font-weight: 900;

        letter-spacing: 1px;

        line-height: 1.1;
    }


    .footer-school {

        font-size: 15px;

        font-weight: 900;

        margin-top: 4px;
    }


    .footer-symbol {

        font-size: 27px;

        margin-top: 4px;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 700px) {

        .block-container {

            padding-left: 12px;

            padding-right: 12px;

            padding-top: 12px;
        }


        .main-title {

            font-size: 31px;
        }


        .main-action {

            min-height: 82px;

            font-size: 17px;
        }


        .sketch-heading {

            font-size: 20px;
        }


        .profile-area {

            right: -2px;
        }


        .footer-tu-es {

            font-size: 26px;
        }


        .footer-school {

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

if "search_text" not in st.session_state:
    st.session_state.search_text = ""

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
# DATENBANK - HINZUFÜGEN
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
# ALLE FUNDSTÜCKE
# ============================================================

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


# ============================================================
# LETZTE FUNDSTÜCKE
# ============================================================

def get_latest_items(limit=6):

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


# ============================================================
# EIN FUNDSTÜCK
# ============================================================

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
# BILD SPEICHERN
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

        pass

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
# KI MODELL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None


    try:

        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

    except Exception:

        return None


# ============================================================
# BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert(
        "RGB"
    )


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
# KI VORHERSAGE
# ============================================================

def predict_image(image):

    model = load_model()


    if model is None:

        return "", 0.0


    try:

        tensor = prepare_image(
            image
        )


        prediction = model(
            tensor,
            training=False,
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

def show_header(
    back=False
):

    st.markdown(
        '<div class="top-header">',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # ZURÜCK
    # --------------------------------------------------------

    if back:

        if st.button(
            "←",
            key="back_button",
        ):

            go("start")


    # --------------------------------------------------------
    # TITEL
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            FUNDBÜRO
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # PROFIL
    # --------------------------------------------------------

    if PROFILE_PATH.exists():

        st.markdown(
            '<div class="profile-area">',
            unsafe_allow_html=True,
        )

        st.image(
            str(PROFILE_PATH),
            width=42,
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="profile-area"
                 style="
                 font-size:30px;
                 font-weight:900;">
                ♙
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# LOGO
# ============================================================

def show_school_logo():

    if not LOGO_PATH.exists():

        return


    st.markdown(
        '<div class="school-logo-area">',
        unsafe_allow_html=True,
    )


    st.image(
        str(LOGO_PATH),
        width=210,
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    st.markdown(
        """
        <div class="sketch-line"></div>
        """,
        unsafe_allow_html=True,
    )


    # Kleine Navigation wie im App-Entwurf

    nav1, nav2, nav3 = st.columns(
        3
    )


    with nav1:

        if st.button(
            "⌂",
            key="footer_start",
        ):

            go("start")


    with nav2:

        if st.button(
            "⌕",
            key="footer_search",
        ):

            go("search")


    with nav3:

        if st.button(
            "♙",
            key="footer_profile",
        ):

            go("profile")


    st.markdown(
        """
        <div class="footer">

            <div class="footer-tu-es">
                TU ES
            </div>

            <div class="footer-school">
                KATHARINEUM ZU LÜBECK
            </div>

            <div class="footer-symbol">
                ∞
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


    # --------------------------------------------------------
    # Immer drei Spalten
    # --------------------------------------------------------

    for start in range(
        0,
        len(items),
        3,
    ):

        row = items[
            start:start + 3
        ]


        columns = st.columns(
            3,
            gap="medium",
        )


        for column, item in zip(
            columns,
            row,
        ):

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
                        <div class="empty-image">
                            KEIN BILD
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


                st.markdown(
                    f"""
                    <div class="item-caption">
                        {name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                if st.button(
                    "ANSEHEN",
                    key=f"show_{item_id}",
                ):

                    st.session_state.selected_item = (
                        item_id
                    )

                    go("detail")


# ============================================================
# STARTSEITE
# ============================================================

def start_page():

    show_school_logo()

    show_header()


    # ========================================================
    # HAUPTAKTIONEN
    # ========================================================

    st.markdown(
        """
        <div class="main-action">
            ↑ &nbsp; FOTO HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    if st.button(
        "FOTO HOCHLADEN",
        key="upload_action",
    ):

        go("upload")


    st.markdown(
        """
        <div class="main-action">
            ⌕ &nbsp; KLEIDUNGSSTÜCK SUCHEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    if st.button(
        "KLEIDUNGSSTÜCK SUCHEN",
        key="search_action",
    ):

        go("search")


    # ========================================================
    # LETZTE FUNDSTÜCKE
    # ========================================================

    st.markdown(
        """
        <div class="sketch-heading">
            LETZTE FUNDSTÜCKE
        </div>
        """,
        unsafe_allow_html=True,
    )


    items = get_latest_items(
        6
    )


    show_item_grid(
        items
    )


    show_footer()


# ============================================================
# UPLOAD
# ============================================================

def upload_page():

    show_header(
        back=True
    )


    st.markdown(
        """
        <div class="sketch-heading">
            FOTO HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # KAMERA
    # ========================================================

    camera = st.camera_input(
        "Foto aufnehmen",
        key="camera",
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
        key="upload",
    )


    file = (
        camera
        if camera is not None
        else uploaded
    )


    if file is None:

        show_footer()

        return


    # ========================================================
    # BILD
    # ========================================================

    image_bytes = file.getvalue()


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


    st.image(
        image,
        width="stretch",
    )


    # ========================================================
    # KI
    # ========================================================

    if (
        st.session_state.last_image_id
        != image_id
    ):

        st.session_state.last_image_id = (
            image_id
        )


        with st.spinner(
            "Bild wird erkannt..."
        ):

            label, confidence = (
                predict_image(image)
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


    # ========================================================
    # DATEN
    # ========================================================

    st.markdown(
        """
        <div class="sketch-heading">
            FUNDSTÜCK
        </div>
        """,
        unsafe_allow_html=True,
    )


    name = st.text_input(
        "Bezeichnung",
        value=label,
        key=f"name_{image_id}",
    )


    kategorie = st.text_input(
        "Kategorie",
        value=label,
        key=f"category_{image_id}",
    )


    farbe = st.text_input(
        "Farbe",
        key=f"color_{image_id}",
    )


    gefunden_am = st.date_input(
        "Gefunden am",
        value=datetime.now().date(),
        key=f"date_{image_id}",
    )


    fundort = st.text_input(
        "Fundort",
        key=f"place_{image_id}",
    )


    notizen = st.text_area(
        "Notizen",
        key=f"notes_{image_id}",
    )


    aktueller_standort = st.text_input(
        "Aktueller Standort",
        value="Fundkiste",
        key=f"location_{image_id}",
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


            st.session_state.last_image_id = None

            st.session_state.ai_result = ""

            st.session_state.ai_confidence = 0.0


            go("start")


    show_footer()


# ============================================================
# SUCHSEITE
# ============================================================

def search_page():

    show_header(
        back=True
    )


    # ========================================================
    # SUCHFELD
    # ========================================================

    st.markdown(
        """
        <div class="search-label">
            ⌕ &nbsp; SUCHEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    search = st.text_input(
        "Suche",
        placeholder="(z.B.:) Flaschen",
        value=st.session_state.search_text,
        label_visibility="collapsed",
        key="search_field",
    )


    st.session_state.search_text = (
        search
    )


    # ========================================================
    # ERGEBNISSE
    # ========================================================

    if search.strip():

        query = search.strip().lower()


        all_items = get_all_items()


        filtered = []


        for item in all_items:

            searchable = " ".join(
                [
                    str(item[2] or ""),
                    str(item[3] or ""),
                    str(item[4] or ""),
                    str(item[6] or ""),
                    str(item[7] or ""),
                    str(item[8] or ""),
                ]
            ).lower()


            if query in searchable:

                filtered.append(
                    item
                )


        st.markdown(
            f"""
            <div class="sketch-heading">
                ERGEBNISSE FÜR
                „{search.upper()}“
            </div>
            """,
            unsafe_allow_html=True,
        )


        show_item_grid(
            filtered
        )


    else:

        st.markdown(
            """
            <div class="sketch-heading">
                ERGEBNISSE
            </div>
            """,
            unsafe_allow_html=True,
        )


        show_item_grid(
            get_all_items()
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
    # SUCHFELD WIE IN DER SKIZZE
    # ========================================================

    st.markdown(
        """
        <div class="search-label">
            ⌕ &nbsp; SUCHEN
        </div>
        """,
        unsafe_allow_html=True,
    )


    detail_search = st.text_input(
        "Detailsuche",
        placeholder="(z.B.:) Flaschen",
        label_visibility="collapsed",
        key="detail_search",
    )


    if detail_search.strip():

        st.session_state.search_text = (
            detail_search
        )

        go("search")

        return


    # ========================================================
    # ERGEBNISÜBERSCHRIFT
    # ========================================================

    st.markdown(
        """
        <div class="sketch-heading">
            FUNDBÜRO
        </div>
        """,
        unsafe_allow_html=True,
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
    # GROSSES BILD
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
    # BEZEICHNUNG
    # ========================================================

    st.markdown(
        f"""
        <div class="sketch-heading">
            {name}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # DETAILZEILEN
    # ========================================================

    st.markdown(
        f"""
        <div class="detail-row">

            <div class="detail-label">
                GEFUNDEN AM:
            </div>

            {gefunden_am}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="detail-row">

            <div class="detail-label">
                FUNDORT:
            </div>

            {fundort or "Keine Angabe"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="detail-row">

            <div class="detail-label">
                NOTIZEN:
            </div>

            {notizen or "Keine Notizen"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
        <div class="detail-row">

            <div class="detail-label">
                AKTUELLER STANDORT:
            </div>

            {aktueller_standort or "Keine Angabe"}

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # ZUSÄTZLICHE INFORMATIONEN
    # ========================================================

    if kategorie or farbe:

        st.markdown(
            f"""
            <div class="detail-row">

                <div class="detail-label">
                    WEITERE ANGABEN:
                </div>

                {kategorie or ""}
                {" · " if kategorie and farbe else ""}
                {farbe or ""}

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # ZUGEORDNET
    # ========================================================

    if zugeordnet:

        st.markdown(
            """
            <div class="assigned">
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
        <div class="sketch-heading">
            PROFIL
        </div>
        """,
        unsafe_allow_html=True,
    )


    if PROFILE_PATH.exists():

        st.image(
            str(PROFILE_PATH),
            width=80,
        )


    if LOGO_PATH.exists():

        show_school_logo()


    st.markdown(
        """
        <div class="detail-row">

            <div class="detail-label">
                KATHARINEUM ZU LÜBECK
            </div>

            Digitales Fundbüro der Schule.

        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="detail-row">

            <div class="detail-label">
                FUNKTIONEN
            </div>

            Foto hochladen
            <br>
            Fundstück fotografieren
            <br>
            automatische Erkennung
            <br>
            Fundstücke suchen
            <br>
            Fundstücke anzeigen
            <br>
            Fundstücke zuordnen

        </div>
        """,
        unsafe_allow_html=True,
    )


    show_footer()


# ============================================================
# APP-ROUTER
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
