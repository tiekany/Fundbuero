# ============================================================
# FUNDBÜRO – KATHARINEUM ZU LÜBECK
# ============================================================

import os

# Wichtig: VOR TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# STREAMLIT
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

LOGO_PATH = ASSETS_DIR / "katharineum_logo.png"

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

DATABASE_PATH = DATA_DIR / "fundbuero.db"


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GRUNDLAYOUT
       ========================= */

    .stApp {
        background: #f3f3f3;
    }

    .main .block-container {
        max-width: 460px;
        padding-top: 12px;
        padding-left: 12px;
        padding-right: 12px;
        padding-bottom: 110px;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* =========================
       SCHRIFT
       ========================= */

    h1, h2, h3 {
        color: #111111 !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-weight: 800 !important;
    }

    h2 {
        text-align: center;
        font-size: 23px !important;
        margin-top: 2px !important;
        margin-bottom: 18px !important;
    }

    h3 {
        font-size: 17px !important;
        margin-top: 18px !important;
        margin-bottom: 10px !important;
    }

    p, label {
        font-family: Arial, Helvetica, sans-serif !important;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        width: 100%;
        min-height: 48px;

        background: #ffffff !important;
        color: #111111 !important;

        border: 2px solid #111111 !important;
        border-radius: 14px !important;

        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #e8e8e8 !important;
        color: #111111 !important;
        border-color: #111111 !important;
    }


    /* =========================
       SUCHFELD
       ========================= */

    .stTextInput input {
        background: #ffffff !important;
        color: #111111 !important;

        border: 2px solid #111111 !important;
        border-radius: 14px !important;

        font-size: 15px !important;
    }

    .stTextInput label {
        display: none;
    }


    /* =========================
       TEXTAREA
       ========================= */

    .stTextArea textarea {
        background: #ffffff !important;
        color: #111111 !important;

        border: 2px solid #111111 !important;
        border-radius: 14px !important;
    }


    /* =========================
       SELECTBOX
       ========================= */

    .stSelectbox > div > div {
        background: #ffffff !important;
        border-radius: 14px !important;
    }


    /* =========================
       INFO BOX
       ========================= */

    .info-box {
        background: #ffffff;
        border: 2px solid #111111;
        border-radius: 14px;

        padding: 14px;
        margin-top: 8px;
        margin-bottom: 12px;

        color: #111111;
        font-family: Arial, Helvetica, sans-serif;
    }


    /* =========================
       DETAIL
       ========================= */

    .detail-label {
        font-size: 12px;
        font-weight: 800;
        color: #111111;

        margin-top: 8px;
    }

    .detail-value {
        font-size: 14px;
        color: #333333;

        margin-bottom: 10px;
    }


    /* =========================
       ZUGEORDNET
       ========================= */

    .assigned {
        background: #ffffff;

        border: 2px solid #111111;
        border-radius: 14px;

        padding: 14px;

        text-align: center;

        font-size: 14px;
        font-weight: 800;

        margin-top: 15px;
    }


    /* =========================
       ABSTAND
       ========================= */

    .bottom-space {
        height: 45px;
    }


    /* =========================
       BILDER
       ========================= */

    [data-testid="stImage"] img {
        border-radius: 10px;
    }


    /* =========================
       MOBILE
       ========================= */

    @media (max-width: 600px) {

        .main .block-container {
            padding-left: 9px;
            padding-right: 9px;
        }

        h2 {
            font-size: 21px !important;
        }

        .stButton > button {
            min-height: 46px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
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
    # 1. Flasche
    # 2) Tasche
    # 3 - Rucksack

    text = re.sub(
        r"^\s*\d+\s*[\.\)\-:\s]+\s*",
        "",
        text
    )

    return text.strip()


# ============================================================
# LABELS LADEN
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
        ) as file:

            for line in file:

                label = clean_label(line)

                if label:
                    labels.append(label)

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

def get_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def create_database():

    connection = get_database()

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

    connection = get_database()

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
# FUNDSTÜCKE LADEN
# ============================================================

def get_items():

    connection = get_database()

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
# EIN FUNDSTÜCK LADEN
# ============================================================

def get_item(item_id):

    connection = get_database()

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
# FUNDSTÜCK ZUORDNEN
# ============================================================

def assign_item(item_id):

    connection = get_database()

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

        image = Image.open(
            file
        ).convert("RGB")

        filename = (
            uuid.uuid4().hex
            + ".jpg"
        )

        path = IMAGE_DIR / filename

        image.save(
            path,
            "JPEG",
            quality=90
        )

        return str(path)

    except Exception as error:

        st.error(
            f"Bild konnte nicht gespeichert werden: {error}"
        )

        return None


# ============================================================
# SCHNELLES KI-MODELL
# ============================================================

@st.cache_resource
def load_fast_model():

    if not MODEL_PATH.exists():

        return None

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # ----------------------------------------------------
        # MODELL AUFWÄRMEN
        # ----------------------------------------------------

        dummy = tf.zeros(
            (1, 224, 224, 3),
            dtype=tf.float32
        )

        model(
            dummy,
            training=False
        )

        return model

    except Exception as error:

        print(
            "KI-Modell konnte nicht geladen werden:"
        )

        print(error)

        return None


# ============================================================
# SCHNELLE PREDICT-FUNKTION
# ============================================================

@st.cache_resource
def get_predict_function():

    model = load_fast_model()

    if model is None:
        return None

    @tf.function(
        reduce_retracing=True
    )
    def predict_function(image):

        return model(
            image,
            training=False
        )

    return predict_function


# ============================================================
# BILD FÜR KI VORBEREITEN
# ============================================================

def prepare_image_fast(image):

    # RGB
    image = image.convert(
        "RGB"
    )

    # Exakt Modellgröße
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    # Tensor
    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    # Teachable Machine Normalisierung
    image = (
        image / 127.5
    ) - 1.0

    # Batch
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# KI ERKENNUNG
# ============================================================

def predict_fast(image):

    predict_function = (
        get_predict_function()
    )

    if predict_function is None:

        return None

    try:

        # Bild vorbereiten
        input_tensor = (
            prepare_image_fast(image)
        )

        # KI
        result = predict_function(
            input_tensor
        )

        probabilities = result[0]

        # Beste Klasse
        index = int(
            tf.argmax(
                probabilities
            ).numpy()
        )

        # Sicherheit
        confidence = float(
            probabilities[index].numpy()
        )

        # Name
        if index < len(LABELS):

            label = LABELS[index]

        else:

            label = (
                f"Klasse {index + 1}"
            )

        return {
            "label": clean_label(label),
            "confidence": confidence
        }

    except Exception as error:

        print(
            "KI-Fehler:"
        )

        print(error)

        return None


# ============================================================
# HEADER
# ============================================================

def show_header():

    left, center, right = st.columns(
        [1, 4, 1]
    )

    # --------------------------------------------------------
    # ZURÜCK
    # --------------------------------------------------------

    with left:

        if st.session_state.page != "start":

            if st.button(
                "←",
                key="header_back",
                width="content"
            ):

                go("start")

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------

    with center:

        if LOGO_PATH.exists():

            st.image(
                str(LOGO_PATH),
                width=58
            )

    # --------------------------------------------------------
    # PROFIL
    # --------------------------------------------------------

    with right:

        if st.button(
            "●",
            key="header_profile",
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

def show_footer():

    st.markdown(
        '<div class="bottom-space"></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    nav1, nav2, nav3 = st.columns(
        3,
        gap="small"
    )

    with nav1:

        if st.button(
            "⌂  START",
            key="bottom_start",
            width="stretch"
        ):

            go("start")

    with nav2:

        if st.button(
            "⌕  SUCHE",
            key="bottom_search",
            width="stretch"
        ):

            go("search")

    with nav3:

        if st.button(
            "●  PROFIL",
            key="bottom_profile",
            width="stretch"
        ):

            go("profile")

    st.write("")

    # --------------------------------------------------------
    # TU ES / KATHARINEUM / LOGO
    # --------------------------------------------------------

    left, middle, right = st.columns(
        [1, 2.7, 1],
        gap="small"
    )

    with left:

        st.markdown(
            """
            <div style="
                text-align:center;
                font-size:17px;
                font-weight:900;
                line-height:0.85;
                color:#111111;
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
                font-size:9px;
                font-weight:900;
                line-height:1;
                margin-top:8px;
                color:#111111;
                white-space:nowrap;
            ">
                KATHARINEUM<br>
                ZU LÜBECK
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        if LOGO_PATH.exists():

            st.image(
                str(LOGO_PATH),
                width=48
            )


# ============================================================
# STARTSEITE
# ============================================================

def start_page():

    show_header()

    st.write("")

    # --------------------------------------------------------
    # FOTO HOCHLADEN
    # --------------------------------------------------------

    if st.button(
        "↑   FOTO HOCHLADEN",
        key="start_upload",
        width="stretch"
    ):

        go("upload")

    st.write("")

    # --------------------------------------------------------
    # SUCHEN
    # --------------------------------------------------------

    if st.button(
        "⌕   FUNDSTÜCK SUCHEN",
        key="start_search",
        width="stretch"
    ):

        go("search")

    # --------------------------------------------------------
    # LETZTE FUNDSTÜCKE
    # --------------------------------------------------------

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

                title = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )

                title = clean_label(
                    title
                )

                if st.button(
                    title,
                    key=f"start_item_{item['id']}",
                    width="stretch"
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go("detail")

    show_footer()


# ============================================================
# UPLOAD-SEITE
# ============================================================

def upload_page():

    show_header()

    st.markdown(
        "### FUNDSTÜCK AUFNEHMEN"
    )

    # --------------------------------------------------------
    # KAMERA
    # --------------------------------------------------------

    camera = st.camera_input(
        "Foto aufnehmen"
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded = st.file_uploader(
        "Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    file = camera if camera else uploaded

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

        detected = st.session_state.ai_result

        # ----------------------------------------------------
        # KI BUTTON
        # ----------------------------------------------------

        if st.button(
            "KI-ERKENNUNG STARTEN",
            key="start_ai",
            width="stretch"
        ):

            with st.spinner(
                "Fundstück wird erkannt..."
            ):

                result = predict_fast(
                    image
                )

            if result:

                detected = result["label"]

                confidence = (
                    result["confidence"]
                    * 100
                )

                # Ergebnis speichern
                st.session_state.ai_result = (
                    detected
                )

                st.markdown(
                    f"""
                    <div class="info-box">

                    <b>Erkannt:</b>
                    {detected}

                    <br>

                    <b>Sicherheit:</b>
                    {confidence:.1f} %

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.error(
                    "Die KI konnte das Fundstück nicht erkennen."
                )

        # ----------------------------------------------------
        # BEREITS ERKANNT
        # ----------------------------------------------------

        elif detected:

            st.markdown(
                f"""
                <div class="info-box">
                <b>Erkannt:</b> {detected}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # INFORMATIONEN
        # ====================================================

        st.markdown(
            "### INFORMATIONEN"
        )

        name = st.text_input(
            "Bezeichnung",
            value=detected,
            key="item_name"
        )

        category = st.selectbox(
            "Kategorie",
            [
                "",
                "Flasche",
                "Tasche",
                "Kleidung",
                "Schuhe",
                "Schulsachen",
                "Elektronik",
                "Sonstiges"
            ],
            key="item_category"
        )

        color = st.text_input(
            "Farbe",
            placeholder="z. B. schwarz",
            key="item_color"
        )

        found_date = st.date_input(
            "Gefunden am",
            key="item_date"
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Obere Turnhalle",
            key="item_location"
        )

        notes = st.text_area(
            "Notizen",
            placeholder="Weitere Informationen...",
            key="item_notes"
        )

        current_location = st.text_input(
            "Aktueller Standort",
            value="Fundkiste",
            key="item_current_location"
        )

        # ====================================================
        # SPEICHERN
        # ====================================================

        if st.button(
            "FUNDSTÜCK SPEICHERN",
            key="save_item",
            width="stretch"
        ):

            image_path = save_image(
                file
            )

            if image_path:

                item_id = save_item(
                    image_path=image_path,
                    name=name,
                    category=category,
                    color=color,
                    found_date=str(found_date),
                    location=location,
                    notes=notes,
                    current_location=current_location
                )

                st.session_state.selected_item = (
                    item_id
                )

                st.session_state.ai_result = ""

                go("detail")

    show_footer()


# ============================================================
# SUCHSEITE
# ============================================================

def search_page():

    show_header()

    # --------------------------------------------------------
    # SUCHFELD
    # --------------------------------------------------------

    search = st.text_input(
        "Suche",
        value=st.session_state.search,
        placeholder="⌕  z. B. Flasche",
        key="search_field"
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

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    if search:

        search_lower = (
            search.lower().strip()
        )

        filtered = []

        for item in items:

            searchable_text = " ".join(
                [
                    item["name"] or "",
                    item["kategorie"] or "",
                    item["farbe"] or "",
                    item["fundort"] or "",
                    item["notizen"] or "",
                    item["aktueller_standort"] or ""
                ]
            ).lower()

            if search_lower in searchable_text:

                filtered.append(item)

    else:

        filtered = items

    # --------------------------------------------------------
    # ERGEBNISSE
    # --------------------------------------------------------

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

                title = (
                    item["name"]
                    or item["kategorie"]
                    or "Fundstück"
                )

                title = clean_label(
                    title
                )

                if st.button(
                    title,
                    key=f"search_item_{item['id']}",
                    width="stretch"
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go("detail")

    show_footer()


# ============================================================
# DETAILSEITE
# ============================================================

def detail_page():

    show_header()

    item = get_item(
        st.session_state.selected_item
    )

    if item is None:

        go("search")

    # --------------------------------------------------------
    # SUCHFELD
    # --------------------------------------------------------

    st.text_input(
        "Suche",
        value=st.session_state.search,
        placeholder="⌕  z. B. Flasche",
        key="detail_search"
    )

    # --------------------------------------------------------
    # BILD
    # --------------------------------------------------------

    image_path = Path(
        item["image_path"]
    )

    if image_path.exists():

        left, center, right = st.columns(
            [1, 3, 1]
        )

        with center:

            st.image(
                str(image_path),
                width="stretch"
            )

    # --------------------------------------------------------
    # TITEL
    # --------------------------------------------------------

    title = (
        item["name"]
        or item["kategorie"]
        or "Fundstück"
    )

    st.markdown(
        f"### {clean_label(title).upper()}"
    )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ZUORDNUNG
    # --------------------------------------------------------

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
            key="assign_item",
            width="stretch"
        ):

            assign_item(
                item["id"]
            )

            st.rerun()

    show_footer()


# ============================================================
# PROFIL
# ============================================================

def profile_page():

    show_header()

    st.markdown(
        "### PROFIL"
    )

    st.markdown(
        """
        <div class="info-box">

        <b>Katharineum zu Lübeck</b>

        <br><br>

        FUNDBÜRO

        <br>

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

    show_footer()


# ============================================================
# SEITENSTEUERUNG
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
