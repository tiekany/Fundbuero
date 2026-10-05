import os
import re
import json
import shutil
import sqlite3
import uuid
import hashlib
import time
from datetime import date, datetime
from pathlib import Path

# ============================================================
# TENSORFLOW
# ============================================================

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# SEITE
# ============================================================

st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon="🔎",
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

for folder in [
    ASSETS_DIR,
    MODEL_DIR,
    IMAGE_DIR,
    DATA_DIR,
]:
    folder.mkdir(exist_ok=True)


# ============================================================
# DATEIEN
# ============================================================

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

DATABASE_PATH = DATA_DIR / "fundbuero.db"
FIXED_MODEL_PATH = DATA_DIR / "keras_model_fixed.h5"

LOGO_PATH = ASSETS_DIR / "katharineum_logo.png"
PROFILE_PATH = ASSETS_DIR / "profile.png"

IMAGE_SIZE = (224, 224)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f3f3f3 !important;
    }

    [data-testid="stHeader"],
    footer {
        display: none;
    }

    .main .block-container {
        max-width: 470px;
        padding-top: 18px;
        padding-left: 12px;
        padding-right: 12px;
        padding-bottom: 105px;
    }

    * {
        font-family: Arial, Helvetica, sans-serif !important;
    }

    h1, h2, h3, p, label {
        color: #111111 !important;
    }

    h2 {
        font-size: 21px !important;
        font-weight: 900 !important;
    }

    h3 {
        font-size: 17px !important;
        font-weight: 900 !important;
    }

    /* --------------------------------------------------------
       HEADER
       -------------------------------------------------------- */

    .app-header {
        display: grid;
        grid-template-columns: 48px 1fr 48px;
        align-items: center;
        margin-bottom: 18px;
    }

    .app-title {
        text-align: center;
        font-size: 27px;
        font-weight: 900;
    }

    .profile-box {
        width: 34px;
        height: 34px;
        border: 3px solid #111111;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: auto;
        background: white;
        overflow: hidden;
    }

    .profile-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* --------------------------------------------------------
       HAUPTAKTIONEN
       -------------------------------------------------------- */

    .main-action {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-height: 82px;
        border: 3px solid #111111;
        border-radius: 14px;
        background: white;
        color: #111111;
        font-size: 17px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 13px;
    }

    .action-icon {
        font-size: 25px;
        margin-right: 11px;
    }

    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button {
        width: 100% !important;
        min-height: 48px !important;
        border: 3px solid #111111 !important;
        border-radius: 12px !important;
        background: white !important;
        color: #111111 !important;
        font-weight: 900 !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #eeeeee !important;
        color: #111111 !important;
        border-color: #111111 !important;
    }

    /* --------------------------------------------------------
       SUCHFELD
       -------------------------------------------------------- */

    .stTextInput input {
        border: 3px solid #111111 !important;
        border-radius: 13px !important;
        background: white !important;
        color: #111111 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        min-height: 47px !important;
    }

    /* --------------------------------------------------------
       TEXTAREA
       -------------------------------------------------------- */

    .stTextArea textarea {
        border: 3px solid #111111 !important;
        border-radius: 12px !important;
        background: white !important;
        color: #111111 !important;
    }

    /* --------------------------------------------------------
       SELECTBOX
       -------------------------------------------------------- */

    .stSelectbox div[data-baseweb="select"] > div {
        border: 3px solid #111111 !important;
        border-radius: 12px !important;
        background: white !important;
    }

    /* --------------------------------------------------------
       BILDER-UPLOAD
       -------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: white !important;
        border: 3px solid #111111 !important;
        border-radius: 14px !important;
        padding: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: white !important;
        border: 0 !important;
    }

    [data-testid="stImage"] img {
        border-radius: 8px !important;
    }

    /* --------------------------------------------------------
       KI
       -------------------------------------------------------- */

    .ai-result {
        border: 3px solid #111111;
        border-radius: 13px;
        background: white;
        padding: 14px;
        margin: 12px 0 16px;
        text-align: center;
    }

    .ai-title {
        font-size: 14px;
        font-weight: 900;
    }

    .ai-name {
        font-size: 20px;
        font-weight: 900;
        margin-top: 4px;
    }

    .ai-confidence {
        font-size: 14px;
        margin-top: 3px;
    }

    /* --------------------------------------------------------
       DETAIL
       -------------------------------------------------------- */

    .detail-row {
        background: white;
        border: 3px solid #111111;
        border-radius: 11px;
        padding: 11px 13px;
        margin-bottom: 9px;
    }

    .detail-label {
        font-size: 11px;
        font-weight: 900;
        margin-bottom: 3px;
    }

    .detail-value {
        font-size: 14px;
        font-weight: 600;
        line-height: 1.35;
    }

    .assigned {
        border: 3px solid #111111;
        border-radius: 12px;
        background: white;
        padding: 13px;
        text-align: center;
        font-weight: 900;
        margin-top: 13px;
    }

    /* --------------------------------------------------------
       FOOTER
       -------------------------------------------------------- */

    .school-footer {
        margin-top: 30px;
        border-top: 3px solid #111111;
        padding-top: 18px;
        text-align: center;
    }

    .footer-tu-es {
        font-size: 20px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .footer-school {
        font-size: 11px;
        font-weight: 900;
        line-height: 1.05;
        margin-bottom: 10px;
    }

    .footer-logo {
        width: 65px;
        max-height: 65px;
        object-fit: contain;
    }

    /* --------------------------------------------------------
       NAVIGATION UNTEN
       -------------------------------------------------------- */

    .bottom-nav {
        position: fixed;
        z-index: 999999;
        left: 50%;
        transform: translateX(-50%);
        bottom: 8px;
        width: min(446px, calc(100% - 20px));
        background: white;
        border: 3px solid #111111;
        border-radius: 16px;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        padding: 4px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.12);
    }

    .bottom-nav a {
        color: #111111 !important;
        text-decoration: none !important;
        text-align: center;
        padding: 8px 2px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 900;
    }

    .bottom-nav a:hover {
        background: #eeeeee;
    }

    .bottom-icon {
        display: block;
        font-size: 18px;
        line-height: 18px;
    }

    @media (max-width: 600px) {

        .main .block-container {
            max-width: 100%;
            padding-left: 9px;
            padding-right: 9px;
            padding-top: 12px;
        }

        .app-title {
            font-size: 24px;
        }

        .main-action {
            min-height: 76px;
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

defaults = {
    "page": "start",
    "selected_item": None,
    "search_term": "",
    "upload_image_id": None,
    "ai_result": "",
    "ai_confidence": 0.0,
    "ai_results": [],
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# NAVIGATION
# ============================================================

VALID_PAGES = {
    "start",
    "upload",
    "search",
    "profile",
    "detail",
}


def go_to(page):

    if page not in VALID_PAGES:
        page = "start"

    st.session_state.page = page
    st.query_params["page"] = page
    st.rerun()


page_from_url = st.query_params.get("page")

if page_from_url in VALID_PAGES:
    st.session_state.page = page_from_url


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def clean_label(label):

    if label is None:
        return ""

    label = str(label).strip()

    label = re.sub(
        r"^\s*\d+\s*[\.\)\-:\s]+\s*",
        "",
        label,
    )

    return label.strip()


def safe_text(value):

    if value is None:
        return ""

    return str(value).strip()


def image_exists(path):

    return bool(
        path
        and Path(path).exists()
    )


# ============================================================
# LABELS
# ============================================================

def load_labels():

    if not LABELS_PATH.exists():

        return [
            "Flasche",
            "Tasche",
            "Rucksack",
            "Kleidung",
            "Sonstiges",
        ]

    try:

        labels = []

        with open(
            LABELS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = clean_label(line)

                if line:
                    labels.append(line)

        if labels:
            return labels

    except Exception:
        pass

    return [
        "Flasche",
        "Tasche",
        "Rucksack",
        "Kleidung",
        "Sonstiges",
    ]


LABELS = load_labels()


# ============================================================
# DATENBANK
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA busy_timeout = 30000"
    )

    return connection


def initialize_database():

    connection = get_connection()

    try:

        connection.execute(
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

        existing = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(fundstuecke)"
            ).fetchall()
        }

        columns = {
            "image_path": "TEXT",
            "name": "TEXT",
            "kategorie": "TEXT",
            "farbe": "TEXT",
            "gefunden_am": "TEXT",
            "fundort": "TEXT",
            "notizen": "TEXT",
            "aktueller_standort": "TEXT",
            "zugeordnet": "INTEGER DEFAULT 0",
            "erstellt_am": "TEXT",
        }

        for column, column_type in columns.items():

            if column not in existing:

                connection.execute(
                    f"""
                    ALTER TABLE fundstuecke
                    ADD COLUMN {column} {column_type}
                    """
                )

        connection.commit()

    finally:

        connection.close()


initialize_database()


# ============================================================
# DATENBANK-RETRY
# ============================================================

def database_operation(operation):

    last_error = None

    for attempt in range(4):

        try:

            return operation()

        except sqlite3.OperationalError as error:

            last_error = error

            if "locked" not in str(error).lower():
                raise

            time.sleep(
                0.4 * (attempt + 1)
            )

    raise last_error


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

    def operation():

        connection = get_connection()

        try:

            cursor = connection.execute(
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_text(image_path),
                    safe_text(name),
                    safe_text(kategorie),
                    safe_text(farbe),
                    safe_text(gefunden_am),
                    safe_text(fundort),
                    safe_text(notizen),
                    safe_text(aktueller_standort),
                    0,
                    datetime.now().isoformat(),
                ),
            )

            connection.commit()

            return cursor.lastrowid

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    return database_operation(operation)


# ============================================================
# FUNDSTÜCKE LADEN
# ============================================================

def get_all_items():

    connection = get_connection()

    try:

        return connection.execute(
            """
            SELECT
                id,
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
            FROM fundstuecke
            ORDER BY id DESC
            """
        ).fetchall()

    finally:

        connection.close()


def get_item(item_id):

    connection = get_connection()

    try:

        return connection.execute(
            """
            SELECT
                id,
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
            FROM fundstuecke
            WHERE id = ?
            """,
            (item_id,),
        ).fetchone()

    finally:

        connection.close()


# ============================================================
# ZUORDNEN
# ============================================================

def update_assigned(item_id):

    def operation():

        connection = get_connection()

        try:

            connection.execute(
                """
                UPDATE fundstuecke
                SET zugeordnet = 1
                WHERE id = ?
                """,
                (item_id,),
            )

            connection.commit()

        except Exception:

            connection.rollback()
            raise

        finally:

            connection.close()

    database_operation(operation)


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
            quality=92,
        )

        return str(path)

    except Exception as error:

        st.error(
            f"Bild konnte nicht gespeichert werden: {error}"
        )

        return None


# ============================================================
# MODELL REPARIEREN
# ============================================================

def repair_h5_model():

    if not MODEL_PATH.exists():
        return None

    if (
        FIXED_MODEL_PATH.exists()
        and FIXED_MODEL_PATH.stat().st_mtime
        >= MODEL_PATH.stat().st_mtime
    ):
        return FIXED_MODEL_PATH

    try:

        import h5py

        shutil.copy2(
            MODEL_PATH,
            FIXED_MODEL_PATH,
        )

        with h5py.File(
            FIXED_MODEL_PATH,
            "r+",
        ) as file:

            if "model_config" not in file.attrs:
                return FIXED_MODEL_PATH

            config = file.attrs[
                "model_config"
            ]

            if isinstance(
                config,
                bytes,
            ):
                config = config.decode(
                    "utf-8"
                )

            config = json.loads(
                config
            )

            def repair(value):

                if isinstance(value, dict):

                    if (
                        value.get("class_name")
                        == "DepthwiseConv2D"
                    ):

                        value.get(
                            "config",
                            {},
                        ).pop(
                            "groups",
                            None,
                        )

                    for child in value.values():
                        repair(child)

                elif isinstance(value, list):

                    for child in value:
                        repair(child)

            repair(config)

            file.attrs[
                "model_config"
            ] = json.dumps(
                config
            ).encode("utf-8")

        return FIXED_MODEL_PATH

    except Exception:

        return None


# ============================================================
# MODELL LADEN
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not MODEL_PATH.exists():

        return (
            None,
            "keras_model.h5 wurde nicht gefunden.",
        )

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        return model, None

    except Exception as first_error:

        fixed = repair_h5_model()

        if fixed is None:

            return None, str(first_error)

        try:

            model = tf.keras.models.load_model(
                fixed,
                compile=False,
            )

            return model, None

        except Exception as error:

            return None, str(error)


# ============================================================
# MODELL FALLBACK
# ============================================================

def run_layers(model, tensor):

    for layer in model.layers:

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


def manual_prediction(model, tensor):

    nested = []

    for layer in model.layers:

        if (
            hasattr(layer, "layers")
            and len(layer.layers) > 0
        ):
            nested.append(layer)

    if len(nested) >= 2:

        tensor = run_layers(
            nested[0],
            tensor,
        )

        tensor = run_layers(
            nested[-1],
            tensor,
        )

        return tensor

    return run_layers(
        model,
        tensor,
    )


# ============================================================
# KI
# ============================================================

def predict_image(image):

    model, error = load_model()

    if model is None:

        raise RuntimeError(
            f"KI-Modell konnte nicht geladen werden:\n{error}"
        )

    image = image.convert("RGB")
    image = image.resize(
        IMAGE_SIZE
    )

    tensor = (
        tf.convert_to_tensor(
            image,
            dtype=tf.float32,
        )
        / 255.0
    )

    tensor = tf.expand_dims(
        tensor,
        0,
    )

    try:

        output = model.predict(
            tensor,
            verbose=0,
        )

    except Exception:

        output = manual_prediction(
            model,
            tensor,
        )

        if hasattr(
            output,
            "numpy",
        ):
            output = output.numpy()

    probabilities = (
        tf.convert_to_tensor(
            output
        )
        .numpy()
        .reshape(-1)
    )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
        or abs(
            float(
                probabilities.sum()
            ) - 1
        ) > 0.05
    ):

        probabilities = (
            tf.nn.softmax(
                probabilities
            )
            .numpy()
        )

    ranking = sorted(
        enumerate(probabilities),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for index, probability in ranking:

        label = (
            LABELS[index]
            if index < len(LABELS)
            else f"Klasse {index + 1}"
        )

        results.append(
            {
                "label": clean_label(label),
                "confidence": float(
                    probability
                ),
            }
        )

    return results


# ============================================================
# HEADER
# ============================================================

def show_header(back=False):

    left, center, right = st.columns(
        [1, 6, 1]
    )

    with left:

        if back:

            if st.button(
                "←",
                key=f"back_{st.session_state.page}",
                width="content",
            ):
                go_to("start")

    with center:

        st.markdown(
            '<div class="app-title">FUNDBÜRO</div>',
            unsafe_allow_html=True,
        )

    with right:

        if PROFILE_PATH.exists():

            st.image(
                PROFILE_PATH,
                width=34,
            )

        else:

            st.markdown(
                """
                <div class="profile-box">
                    <span>●</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    logo = ""

    if LOGO_PATH.exists():

        logo = f"""
        <img
            class="footer-logo"
            src="{LOGO_PATH.as_uri()}"
        >
        """

    st.markdown(
        f"""
        <div class="school-footer">

            <div class="footer-tu-es">
                TU ES
            </div>

            <div class="footer-school">
                KATHARINEUM<br>
                ZU LÜBECK
            </div>

            {logo}

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BOTTOM NAVIGATION
# ============================================================

def show_bottom_navigation():

    st.markdown(
        """
        <div class="bottom-nav">

            <a href="?page=start">
                <span class="bottom-icon">⌂</span>
                START
            </a>

            <a href="?page=search">
                <span class="bottom-icon">⌕</span>
                SUCHEN
            </a>

            <a href="?page=upload">
                <span class="bottom-icon">＋</span>
                FUND
            </a>

            <a href="?page=profile">
                <span class="bottom-icon">○</span>
                PROFIL
            </a>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BILD-GRID
# ============================================================

def image_grid(
    items,
    key_prefix="grid",
):

    if not items:

        st.info(
            "Noch keine Fundstücke vorhanden."
        )

        return

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
            gap="small",
        )

        for index, item in enumerate(row):

            with columns[index]:

                if image_exists(
                    item["image_path"]
                ):

                    st.image(
                        item["image_path"],
                        width="stretch",
                    )

                else:

                    st.markdown(
                        """
                        <div style="
                            height:110px;
                            border:3px solid #111;
                            background:#fff;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-weight:900;
                        ">
                            KEIN BILD
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if st.button(
                    clean_label(
                        item["name"]
                    ) or "FUNDSTÜCK",
                    key=f"{key_prefix}_{item['id']}",
                ):

                    st.session_state.selected_item = (
                        item["id"]
                    )

                    go_to("detail")


# ============================================================
# START
# ============================================================

def page_start():

    show_header()

    st.markdown(
        """
        <div class="main-action">
            <span class="action-icon">⇧</span>
            FOTO HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "FOTO HOCHLADEN",
        key="start_upload",
    ):
        go_to("upload")

    st.markdown(
        """
        <div style="height:7px"></div>

        <div class="main-action">
            <span class="action-icon">⌕</span>
            KLEIDUNGSSTÜCK SUCHEN
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "KLEIDUNGSSTÜCK SUCHEN",
        key="start_search",
    ):
        go_to("search")

    st.markdown(
        "### LETZTE FUNDSTÜCKE"
    )

    items = get_all_items()

    image_grid(
        items[:6],
        "start",
    )

    show_footer()
    show_bottom_navigation()


# ============================================================
# UPLOAD
# ============================================================

def page_upload():

    show_header(back=True)

    st.markdown(
        "### FUNDSTÜCK HOCHLADEN"
    )

    # ========================================================
    # WICHTIG:
    # HIER GIBT ES KEINE KAMERA UND KEIN VIDEO.
    # AUSSCHLIESSLICH BILDDATEIEN.
    # ========================================================

    uploaded_file = st.file_uploader(
        "Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=False,
        key="photo_upload",
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div style="
                text-align:center;
                font-size:13px;
                font-weight:700;
                margin:12px 0 20px;
            ">
                Bitte ein Foto des Fundstücks auswählen.
                <br>
                JPG, JPEG oder PNG
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_footer()
        show_bottom_navigation()

        return

    # ========================================================
    # ZUSÄTZLICHE SICHERHEIT:
    # KEINE VIDEODATEIEN VERARBEITEN
    # ========================================================

    allowed_types = {
        "image/jpeg",
        "image/png",
    }

    if (
        uploaded_file.type
        and uploaded_file.type
        not in allowed_types
    ):

        st.error(
            "Bitte ausschließlich JPG-, JPEG- oder PNG-Bilder hochladen."
        )

        show_footer()
        show_bottom_navigation()

        return

    # ========================================================
    # BILD ÖFFNEN
    # ========================================================

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Die Datei ist kein gültiges Bild."
        )

        show_footer()
        show_bottom_navigation()

        return

    st.image(
        image,
        width="stretch",
    )

    # ========================================================
    # BILD-ID
    # ========================================================

    image_bytes = uploaded_file.getvalue()

    image_id = hashlib.md5(
        image_bytes
    ).hexdigest()

    # ========================================================
    # KI
    # ========================================================

    if (
        st.session_state.upload_image_id
        != image_id
    ):

        st.session_state.upload_image_id = image_id
        st.session_state.ai_result = ""
        st.session_state.ai_confidence = 0.0
        st.session_state.ai_results = []

        try:

            with st.spinner(
                "Fundstück wird erkannt ..."
            ):

                results = predict_image(
                    image
                )

            if results:

                st.session_state.ai_results = (
                    results
                )

                st.session_state.ai_result = (
                    results[0]["label"]
                )

                st.session_state.ai_confidence = (
                    results[0]["confidence"]
                )

        except Exception as error:

            st.error(
                "Die KI-Erkennung konnte nicht durchgeführt werden."
            )

            with st.expander(
                "Technische Fehlermeldung"
            ):

                st.code(
                    str(error)
                )

    # ========================================================
    # KI-ERGEBNIS
    # ========================================================

    if st.session_state.ai_result:

        confidence = round(
            st.session_state.ai_confidence
            * 100
        )

        st.markdown(
            f"""
            <div class="ai-result">

                <div class="ai-title">
                    KI-ERKENNUNG
                </div>

                <div class="ai-name">
                    Erkannt:
                    {st.session_state.ai_result}
                </div>

                <div class="ai-confidence">
                    Sicherheit: {confidence} %
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # FORMULAR
    # ========================================================

    st.markdown(
        "### ANGABEN ZUM FUNDSTÜCK"
    )

    with st.form(
        "save_item_form",
    ):

        name = st.text_input(
            "Bezeichnung",
            value=(
                st.session_state.ai_result
                or ""
            ),
            placeholder="z. B. Trinkflasche",
        )

        farbe = st.text_input(
            "Farbe",
            placeholder="z. B. schwarz",
        )

        fundort = st.text_input(
            "Fundort",
            placeholder="z. B. Obere Turnhalle",
        )

        notizen = st.text_area(
            "Notizen",
            placeholder="Weitere Informationen",
        )

        submitted = st.form_submit_button(
            "FUNDSTÜCK SPEICHERN",
            width="stretch",
        )

    # ========================================================
    # SPEICHERN
    # ========================================================

    if submitted:

        image_path = save_uploaded_image(
            uploaded_file
        )

        if image_path is None:
            return

        try:

            item_id = add_item(
                image_path=image_path,
                name=name or "Fundstück",
                kategorie=(
                    st.session_state.ai_result
                    or "Sonstiges"
                ),
                farbe=farbe,
                gefunden_am=(
                    date.today().strftime(
                        "%d.%m.%Y"
                    )
                ),
                fundort=fundort,
                notizen=notizen,
                aktueller_standort="Fundkiste",
            )

            st.session_state.selected_item = (
                item_id
            )

            st.success(
                "Fundstück wurde gespeichert."
            )

            time.sleep(0.3)

            go_to("detail")

        except sqlite3.OperationalError as error:

            st.error(
                "Das Fundstück konnte nicht gespeichert werden."
            )

            st.code(
                str(error)
            )

        except Exception as error:

            st.error(
                "Beim Speichern ist ein Fehler aufgetreten."
            )

            with st.expander(
                "Technische Fehlermeldung"
            ):

                st.code(
                    str(error)
                )

    show_footer()
    show_bottom_navigation()


# ============================================================
# SUCHE
# ============================================================

def page_search():

    show_header(back=True)

    st.markdown(
        "### FUNDSTÜCK SUCHEN"
    )

    search_term = st.text_input(
        "Suche",
        value=st.session_state.search_term,
        placeholder="z. B. Flaschen",
        key="search_input",
    )

    st.session_state.search_term = (
        search_term
    )

    with st.expander(
        "FILTER"
    ):

        color_filter = st.text_input(
            "Farbe",
        )

        category_filter = st.selectbox(
            "Kategorie",
            [
                "Alle",
                "Flasche",
                "Tasche",
                "Rucksack",
                "Kleidung",
                "Sonstiges",
            ],
        )

    items = get_all_items()

    filtered = []

    search = (
        search_term
        .strip()
        .lower()
    )

    color = (
        color_filter
        .strip()
        .lower()
    )

    for item in items:

        searchable = " ".join(
            [
                safe_text(item["name"]),
                safe_text(item["kategorie"]),
                safe_text(item["farbe"]),
                safe_text(item["fundort"]),
                safe_text(item["notizen"]),
            ]
        ).lower()

        if search and search not in searchable:
            continue

        if (
            color
            and color
            not in safe_text(
                item["farbe"]
            ).lower()
        ):
            continue

        if (
            category_filter != "Alle"
            and category_filter.lower()
            not in searchable
        ):
            continue

        filtered.append(item)

    if search:

        st.markdown(
            f"### ERGEBNISSE FÜR „{search_term.strip()}“"
        )

    else:

        st.markdown(
            "### ALLE FUNDSTÜCKE"
        )

    image_grid(
        filtered,
        "search",
    )

    show_footer()
    show_bottom_navigation()


# ============================================================
# DETAIL
# ============================================================

def page_detail():

    show_header(back=True)

    item_id = (
        st.session_state.selected_item
    )

    if not item_id:

        st.warning(
            "Kein Fundstück ausgewählt."
        )

        return

    item = get_item(
        item_id
    )

    if item is None:

        st.error(
            "Fundstück wurde nicht gefunden."
        )

        return

    st.markdown(
        f"""
        <h2 style="text-align:center;">
            {clean_label(item["name"]) or "FUNDBÜRO"}
        </h2>
        """,
        unsafe_allow_html=True,
    )

    if image_exists(
        item["image_path"]
    ):

        st.image(
            item["image_path"],
            width="stretch",
        )

    st.markdown(
        f"""
        <div class="detail-row">

            <div class="detail-label">
                GEFUNDEN AM
            </div>

            <div class="detail-value">
                {safe_text(item["gefunden_am"])}
            </div>

        </div>

        <div class="detail-row">

            <div class="detail-label">
                FUNDORT
            </div>

            <div class="detail-value">
                {safe_text(item["fundort"]) or "-"}
            </div>

        </div>

        <div class="detail-row">

            <div class="detail-label">
                NOTIZEN
            </div>

            <div class="detail-value">
                {safe_text(item["notizen"]) or "-"}
            </div>

        </div>

        <div class="detail-row">

            <div class="detail-label">
                AKTUELLER STANDORT
            </div>

            <div class="detail-value">
                {safe_text(item["aktueller_standort"]) or "Fundkiste"}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if item["zugeordnet"]:

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

            try:

                update_assigned(
                    item_id
                )

                st.success(
                    "Fundstück wurde zugeordnet."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "Zuordnung konnte nicht gespeichert werden."
                )

                st.code(
                    str(error)
                )

    show_footer()
    show_bottom_navigation()


# ============================================================
# PROFIL
# ============================================================

def page_profile():

    show_header(back=True)

    st.markdown(
        "### PROFIL"
    )

    if PROFILE_PATH.exists():

        st.image(
            PROFILE_PATH,
            width=90,
        )

    st.markdown(
        """
        <div style="
            text-align:center;
            margin:14px 0 20px;
        ">
            <strong>KATHARINEUM ZU LÜBECK</strong>
            <br>
            Fundbüro-App
        </div>
        """,
        unsafe_allow_html=True,
    )

    items = get_all_items()

    st.markdown(
        f"""
        <div class="detail-row">

            <div class="detail-label">
                GESPEICHERTE FUNDSTÜCKE
            </div>

            <div class="detail-value">
                {len(items)}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    show_footer()
    show_bottom_navigation()


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

elif st.session_state.page == "profile":

    page_profile()

else:

    page_start()
    
