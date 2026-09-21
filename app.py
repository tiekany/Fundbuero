import os
import re
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

# ============================================================
# TENSORFLOW
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# EINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon="assets/katharineum_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# ORDNER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ASSETS = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "modell"
IMAGE_DIR = BASE_DIR / "bilder"
DATA_DIR = BASE_DIR / "daten"

ASSETS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# DATEIEN
# ============================================================

LOGO = ASSETS / "katharineum_logo.png"

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

DATABASE = DATA_DIR / "fundbuero.db"
FIXED_MODEL = DATA_DIR / "keras_model_fixed.h5"

IMAGE_SIZE = (224, 224)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       APP
       -------------------------------------------------------- */

    .stApp {
        background: #f2f2f2;
    }

    .main .block-container {
        max-width: 440px;
        padding-top: 15px;
        padding-left: 12px;
        padding-right: 12px;
        padding-bottom: 90px;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* --------------------------------------------------------
       TEXT
       -------------------------------------------------------- */

    h1, h2, h3 {
        color: #111111 !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 800 !important;
    }

    h2 {
        font-size: 24px !important;
        text-align: center;
        margin-top: 4px !important;
        margin-bottom: 18px !important;
    }

    h3 {
        font-size: 17px !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }

    p, label, div {
        font-family: Arial, Helvetica, sans-serif;
    }


    /* --------------------------------------------------------
       LOGO
       -------------------------------------------------------- */

    [data-testid="stImage"] img {
        border-radius: 0 !important;
    }


    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button {
        width: 100%;
        min-height: 48px;

        background: #ffffff !important;
        color: #111111 !important;

        border: 2px solid #111111 !important;
        border-radius: 14px !important;

        font-size: 14px !important;
        font-weight: 700 !important;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #e9e9e9 !important;
        color: #111111 !important;
        border-color: #111111 !important;
    }


    /* --------------------------------------------------------
       SUCHFELD
       -------------------------------------------------------- */

    .stTextInput input {
        background: #ffffff !important;
        color: #111111 !important;

        border: 2px solid #111111 !important;
        border-radius: 14px !important;

        font-size: 15px !important;
        padding: 10px !important;
    }

    .stTextInput label {
        display: none;
    }


    /* --------------------------------------------------------
       TEXTAREA
       -------------------------------------------------------- */

    .stTextArea textarea {
        background: #ffffff !important;
        color: #111111 !important;

        border: 2px solid #111111 !important;
        border-radius: 14px !important;
    }


    /* --------------------------------------------------------
       SELECTBOX
       -------------------------------------------------------- */

    .stSelectbox > div > div {
        background: #ffffff !important;
        border-radius: 14px !important;
    }


    /* --------------------------------------------------------
       FOOTER NAVIGATION
       -------------------------------------------------------- */

    .bottom-space {
        height: 70px;
    }


    /* --------------------------------------------------------
       INFOBOX
       -------------------------------------------------------- */

    .info-box {
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 14px;
        padding: 13px;
        margin-top: 8px;
        margin-bottom: 8px;
    }


    /* --------------------------------------------------------
       DETAIL
       -------------------------------------------------------- */

    .detail-label {
        font-weight: 800;
        font-size: 13px;
        color: #111111;
        margin-bottom: 2px;
    }

    .detail-value {
        font-size: 14px;
        color: #333333;
        margin-bottom: 12px;
    }


    /* --------------------------------------------------------
       STATUS
       -------------------------------------------------------- */

    .assigned {
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 14px;
        padding: 13px;
        text-align: center;
        font-weight: 800;
        margin-top: 15px;
    }


    /* --------------------------------------------------------
       MOBILE
       -------------------------------------------------------- */

    @media (max-width: 600px) {

        .main .block-container {
            padding-left: 9px;
            padding-right: 9px;
        }

        h2 {
            font-size: 22px !important;
        }

        .stButton > button {
            min-height: 46px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

if "search" not in st.session_state:
    st.session_state.search = ""


# ============================================================
# NAVIGATION
# ============================================================

def go(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# LABEL BEREINIGEN
# ============================================================

def clean_label(text):

    if not text:
        return ""

    text = str(text).strip()

    # Entfernt z.B.
    # 0 Flasche
    # 1 Flasche
    # 2. Tasche
    # 3) Rucksack

    text = re.sub(
        r"^\s*\d+\s*[\.\)\-:\s]+\s*",
        "",
        text
    )

    return text.strip()


# ============================================================
# LABELS
# ============================================================

def load_labels():

    if not LABELS_PATH.exists():
        return [
            "Flasche",
            "Tasche",
            "Kleidung",
            "Schuhe",
            "Sonstiges"
        ]

    labels = []

    try:

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = clean_label(line)

                if line:
                    labels.append(line)

    except Exception:

        return [
            "Flasche",
            "Tasche",
            "Kleidung",
            "Schuhe",
            "Sonstiges"
        ]

    return labels


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def db():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


def create_database():

    connection = db()

    connection.execute(
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


create_database()


# ============================================================
# FUNDSTÜCK SPEICHERN
# ============================================================

def save_item(
    image_path,
    name,
    category,
    color,
    found_date,
    location,
    notes,
    current_location
):

    connection = db()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO fundstuecke
        (
            image_path,
            name,
            kategorie,
            farbe,
            gefunden_am,
            fundort,
            notizen,
            aktueller_standort,
            erstellt_am
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_path,
            clean_label(name),
            clean_label(category),
            color,
            found_date,
            location,
            notes,
            current_location,
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

def get_items():

    connection = db()

    items = connection.execute(
        """
        SELECT *
        FROM fundstuecke
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return items


# ============================================================
# EIN FUNDSTÜCK
# ============================================================

def get_item(item_id):

    connection = db()

    item = connection.execute(
        """
        SELECT *
        FROM fundstuecke
        WHERE id = ?
        """,
        (item_id,)
    ).fetchone()

    connection.close()

    return item


# ============================================================
# ZUORDNEN
# ============================================================

def assign_item(item_id):

    connection = db()

    connection.execute(
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
# BILD SPEICHERN
# ============================================================

def save_image(file):

    try:

        image = Image.open(file).convert("RGB")

        filename = (
            uuid.uuid4().hex
            + ".jpg"
        )

        path = IMAGE_DIR / filename

        image.save(
            path,
            "JPEG",
            quality=92
        )

        return str(path)

    except Exception as e:

        st.error(
            f"Bild konnte nicht gespeichert werden: {e}"
        )

        return None


# ============================================================
# MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception:

        return None


# ============================================================
# BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    data = tf.keras.utils.img_to_array(
        image
    )

    data = (
        data.astype("float32")
        / 127.5
    ) - 1.0

    return tf.expand_dims(
        data,
        0
    )


# ============================================================
# KI
# ============================================================

def predict(image):

    model = load_model()

    if model is None:
        return None

    try:

        data = prepare_image(
            image
        )

        result = model.predict(
            data,
            verbose=0
        )

        probabilities = (
            result[0]
        )

        index = int(
            probabilities.argmax()
        )

        confidence = float(
            probabilities[index]
        )

        if index < len(LABELS):

            label = LABELS[index]

        else:

            label = f"Klasse {index + 1}"

        return {
            "label": clean_label(label),
            "confidence": confidence
        }

    except Exception:

        return None


# ============================================================
# HEADER
# ============================================================

def header():

    top_left, top_center, top_right = st.columns(
        [1, 4, 1]
    )

    with top_left:

        if st.session_state.page != "start":

            if st.button(
                "←",
                key="back",
                width="content"
            ):

                go("start")

    with top_center:

        if LOGO.exists():

            st.image(
                LOGO,
                width=58
            )

    with top_right:

        if st.button(
            "●",
            key="profile_top",
            width="content"
        ):

            go("profile")

    st.markdown(
        "<h2>FUNDBÜRO</h2>",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

def footer():

    st.markdown(
        '<div class="bottom-space"></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Navigation
    nav1, nav2, nav3 = st.columns(3)

    with nav1:

        if st.button(
            "⌂\nSTART",
            key="nav_start",
            width="stretch"
        ):

            go("start")

    with nav2:

        if st.button(
            "⌕\nSUCHE",
            key="nav_search",
            width="stretch"
        ):

            go("search")

    with nav3:

        if st.button(
            "●\nPROFIL",
            key="nav_profile",
            width="stretch"
        ):

            go("profile")

    st.write("")

    # ========================================================
    # TU ES + SCHRIFT + LOGO
    # ========================================================

    left, middle, right = st.columns(
        [1, 2.7, 1],
        gap="small"
    )

    with left:

        st.markdown(
            """
            <div style="
                text-align:center;
                color:#111111;
                font-size:17px;
                font-weight:900;
                line-height:0.85;
            ">
                TU<br>ES
            </div>
            """,
            unsafe_allow_html=True
        )

    with middle:

        st.markdown(
            """
            <div style="
                text-align:center;
                color:#111111;
                font-size:9px;
                font-weight:900;
                line-height:1.0;
                margin-top:7px;
                white-space:nowrap;
            ">
                KATHARINEUM<br>
                ZU LÜBECK
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        if LOGO.exists():

            st.image(
                LOGO,
                width=48
            )


# ============================================================
# STARTSEITE
# ============================================================

def start_page():

    header()

    # ========================================================
    # FOTO AUFNEHMEN
    # ========================================================

    if st.button(
        "📷   FOTO AUFNEHMEN",
        key="camera",
        width="stretch"
    ):

        go("upload")


    # ========================================================
    # FOTO HOCHLADEN
    # ========================================================

    if st.button(
        "↑   FOTO HOCHLADEN",
        key="upload",
        width="stretch"
    ):

        go("upload")


    # ========================================================
    # SUCHEN
    # ========================================================

    if st.button(
        "⌕   FUNDSTÜCK SUCHEN",
        key="search_start",
        width="stretch"
    ):

        go("search")


    # ========================================================
    # LETZTE FUNDSTÜCKE
    # ========================================================

    st.markdown(
        "### LETZTE FUNDSTÜCKE"
    )

    items = get_items()

    if not items:

        st.info(
            "Noch keine Fundstücke vorhanden."
        )

    else:

        columns = st.columns(
            3,
            gap="small"
        )

        for i, item in enumerate(
            items[:6]
        ):

            with columns[
                i % 3
            ]:

                path = Path(
                    item["image_path"]
                )

                if path.exists():

                    st.image(
                        str(path),
                        width="stretch"
                    )

                name = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )

                name = clean_label(
                    name
                )

                if st.button(
                    name,
                    key=f"item_{item['id']}",
                    width="stretch"
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go("detail")

    footer()


# ============================================================
# UPLOAD-SEITE
# ============================================================

def upload_page():

    header()

    st.markdown(
        "### FUNDSTÜCK AUFNEHMEN"
    )

    camera = st.camera_input(
        "Foto aufnehmen"
    )

    uploaded = st.file_uploader(
        "Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    file = camera or uploaded

    if file:

        image = Image.open(
            file
        ).convert("RGB")

        st.image(
            image,
            width="stretch"
        )

        # ====================================================
        # KI
        # ====================================================

        st.markdown(
            "### KI-ERKENNUNG"
        )

        result = predict(
            image
        )

        detected = ""

        if result:

            detected = result["label"]

            percentage = (
                result["confidence"]
                * 100
            )

            st.markdown(
                f"""
                <div class="info-box">
                <b>Erkannt:</b> {detected}<br>
                <b>Sicherheit:</b> {percentage:.1f} %
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "KI-Erkennung konnte nicht durchgeführt werden."
            )

        # ====================================================
        # DATEN
        # ====================================================

        st.markdown(
            "### INFORMATIONEN"
        )

        name = st.text_input(
            "Bezeichnung",
            value=detected
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

        if st.button(
            "FUNDSTÜCK SPEICHERN",
            key="save",
            width="stretch"
        ):

            image_path = save_image(
                file
            )

            if image_path:

                item_id = save_item(
                    image_path,
                    name,
                    category,
                    color,
                    str(found_date),
                    location,
                    notes,
                    current_location
                )

                st.session_state.selected_item = (
                    item_id
                )

                go("detail")

    footer()


# ============================================================
# SUCHSEITE
# ============================================================

def search_page():

    header()

    # ========================================================
    # SUCHFELD
    # ========================================================

    search = st.text_input(
        "Suche",
        value=st.session_state.search,
        placeholder="⌕  z. B. Flasche",
        key="search_input"
    )

    st.session_state.search = search

    if search:

        st.markdown(
            f"### ERGEBNISSE FÜR „{search.upper()}“"
        )

    else:

        st.markdown(
            "### ALLE FUNDSTÜCKE"
        )

    items = get_items()

    search_lower = search.lower().strip()

    if search_lower:

        filtered = []

        for item in items:

            text = " ".join(
                [
                    item["name"] or "",
                    item["kategorie"] or "",
                    item["farbe"] or "",
                    item["fundort"] or "",
                    item["notizen"] or ""
                ]
            ).lower()

            if search_lower in text:

                filtered.append(item)

    else:

        filtered = items

    # ========================================================
    # RASTER
    # ========================================================

    if not filtered:

        st.info(
            "Keine Fundstücke gefunden."
        )

    else:

        columns = st.columns(
            3,
            gap="small"
        )

        for i, item in enumerate(
            filtered
        ):

            with columns[
                i % 3
            ]:

                path = Path(
                    item["image_path"]
                )

                if path.exists():

                    st.image(
                        str(path),
                        width="stretch"
                    )

                name = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )

                name = clean_label(
                    name
                )

                if st.button(
                    name,
                    key=f"search_{item['id']}",
                    width="stretch"
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go("detail")

    footer()


# ============================================================
# DETAILSEITE
# ============================================================

def detail_page():

    header()

    item = get_item(
        st.session_state.selected_item
    )

    if item is None:

        go("search")

    # ========================================================
    # SUCHFELD
    # ========================================================

    st.text_input(
        "Suche",
        value=st.session_state.search,
        placeholder="⌕  z. B. Flasche",
        key="detail_search"
    )

    # ========================================================
    # BILD
    # ========================================================

    path = Path(
        item["image_path"]
    )

    if path.exists():

        image_columns = st.columns(
            [1, 3, 1]
        )

        with image_columns[1]:

            st.image(
                str(path),
                width="stretch"
            )

    # ========================================================
    # TITEL
    # ========================================================

    name = (
        item["name"]
        or item["kategorie"]
        or "Fundstück"
    )

    st.markdown(
        f"### {clean_label(name).upper()}"
    )

    # ========================================================
    # DETAILS
    # ========================================================

    st.markdown(
        f"""
        <div class="info-box">

        <div class="detail-label">
        GEFUNDEN AM:
        </div>

        <div class="detail-value">
        {item["gefunden_am"] or "-"}
        </div>


        <div class="detail-label">
        FUNDORT:
        </div>

        <div class="detail-value">
        {item["fundort"] or "-"}
        </div>


        <div class="detail-label">
        NOTIZEN:
        </div>

        <div class="detail-value">
        {item["notizen"] or "-"}
        </div>


        <div class="detail-label">
        AKTUELLER STANDORT:
        </div>

        <div class="detail-value">
        {item["aktueller_standort"] or "-"}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # ZUORDNUNG
    # ========================================================

    if item["zugeordnet"]:

        st.markdown(
            """
            <div class="assigned">
            FUNDSTÜCK ZUGEORDNET
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        if st.button(
            "FUNDSTÜCK ZUORDNEN",
            key="assign",
            width="stretch"
        ):

            assign_item(
                item["id"]
            )

            st.rerun()

    footer()


# ============================================================
# PROFIL
# ============================================================

def profile_page():

    header()

    st.markdown(
        "### PROFIL"
    )

    st.markdown(
        """
        <div class="info-box">

        <b>Katharineum zu Lübeck</b><br><br>

        Fundbüro-App<br>
        Digitale Verwaltung von Fundstücken

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "← ZURÜCK",
        key="profile_back",
        width="stretch"
    ):

        go("start")

    footer()


# ============================================================
# APP
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

    st.session_state.page = "start"

    start_page()
