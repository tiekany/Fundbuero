import os
import re
import sqlite3
import hashlib
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# GRUNDLAGEN
# ---------------------------------------------------------

BASE = Path(__file__).parent
ASSETS = BASE / "assets"
MODEL_DIR = BASE / "modell"
IMAGES = BASE / "bilder"
DATA = BASE / "daten"

DB = DATA / "fundbuero.db"
MODEL_FILE = MODEL_DIR / "keras_model.h5"
LABEL_FILE = MODEL_DIR / "labels.txt"

IMAGES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------------
# DESIGN
# ---------------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', Arial, sans-serif;
}

.stApp {
    background: #f3f3f3;
    color: #000;
}

.block-container {
    max-width: 950px;
    padding-top: 25px;
    padding-bottom: 130px;
}

/* Allgemeine Buttons */

.stButton > button {
    width: 100%;
    border: 3px solid #000;
    border-radius: 14px;
    background: #fff;
    color: #000;
    font-weight: 600;
    min-height: 48px;
    transition: 0.15s;
}

.stButton > button:hover {
    background: #000;
    color: #fff;
    border-color: #000;
}

/* Kopf */

.app-header {
    display: grid;
    grid-template-columns: 50px 1fr 50px;
    align-items: center;
    margin-bottom: 25px;
}

.logo-title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 2px;
}

.profile-img {
    width: 38px;
    height: 38px;
    border: 2px solid #000;
    border-radius: 50%;
    object-fit: cover;
}

/* Überschriften */

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin: 28px 0 14px 0;
    letter-spacing: .5px;
}

/* Große Startseiten-Felder */

.main-action {
    background: #fff;
    border: 3px solid #000;
    border-radius: 16px;
    min-height: 105px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 14px;
}

.action-icon {
    font-size: 28px;
    margin-right: 12px;
}

/* Bilder */

.image-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.grid-image {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border: 3px solid #000;
    background: #fff;
}

/* Suchfeld */

.search-box {
    background: #fff;
    border: 3px solid #000;
    border-radius: 14px;
    padding: 4px 10px;
    margin-bottom: 20px;
}

/* Detailinformationen */

.info-box {
    background: #fff;
    border: 3px solid #000;
    border-radius: 14px;
    padding: 15px;
    margin-top: 10px;
}

.info-label {
    font-weight: 700;
    margin-bottom: 3px;
}

.info-text {
    margin-bottom: 12px;
}

/* KI */

.ai-box {
    background: #fff;
    border: 3px solid #000;
    border-radius: 14px;
    padding: 15px;
    margin: 18px 0;
}

.ai-title {
    font-weight: 700;
    font-size: 17px;
}

/* Zugeordnet */

.assigned {
    border: 3px solid #000;
    border-radius: 14px;
    background: #fff;
    padding: 15px;
    text-align: center;
    font-weight: 700;
    margin-top: 15px;
}

/* Footer */

.footer {
    margin-top: 45px;
    border-top: 3px solid #000;
    padding-top: 18px;
    text-align: center;
}

.footer-tu-es {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 2px;
}

.footer-school {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 3px;
}

.footer-symbol {
    font-size: 28px;
    margin-top: 5px;
}

/* Mobile */

@media (max-width: 600px) {

    .block-container {
        padding-left: 14px;
        padding-right: 14px;
    }

    .logo-title {
        font-size: 24px;
    }

    .main-action {
        min-height: 90px;
        font-size: 17px;
    }

    .image-grid {
        gap: 8px;
    }

}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATENBANK
# ---------------------------------------------------------

def db():
    return sqlite3.connect(DB)


def init_db():
    con = db()

    con.execute("""
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
            zugeordnet INTEGER DEFAULT 0
        )
    """)

    con.commit()
    con.close()


init_db()

# ---------------------------------------------------------
# DATENBANK-FUNKTIONEN
# ---------------------------------------------------------

def get_items(search=None):
    con = db()

    if search:
        like = f"%{search}%"

        rows = con.execute("""
            SELECT *
            FROM fundstuecke
            WHERE name LIKE ?
               OR kategorie LIKE ?
               OR farbe LIKE ?
               OR fundort LIKE ?
               OR notizen LIKE ?
               OR aktueller_standort LIKE ?
            ORDER BY id DESC
        """, (like, like, like, like, like, like)).fetchall()

    else:
        rows = con.execute("""
            SELECT *
            FROM fundstuecke
            ORDER BY id DESC
        """).fetchall()

    con.close()
    return rows


def get_item(item_id):
    con = db()

    row = con.execute("""
        SELECT *
        FROM fundstuecke
        WHERE id = ?
    """, (item_id,)).fetchone()

    con.close()
    return row


def add_item(
    image_path,
    name,
    kategorie="",
    farbe="",
    gefunden_am="",
    fundort="",
    notizen="",
    aktueller_standort="Fundkiste"
):
    con = db()

    con.execute("""
        INSERT INTO fundstuecke
        (
            image_path,
            name,
            kategorie,
            farbe,
            gefunden_am,
            fundort,
            notizen,
            aktueller_standort
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_path,
        name,
        kategorie,
        farbe,
        gefunden_am,
        fundort,
        notizen,
        aktueller_standort
    ))

    con.commit()
    con.close()

# ---------------------------------------------------------
# LABELS
# ---------------------------------------------------------

def clean_label(label):
    label = label.strip()

    # z.B. "0 Flasche" -> "Flasche"
    label = re.sub(r"^\s*\d+\s*[\)\.\-:]?\s*", "", label)

    return label


def load_labels():
    if not LABEL_FILE.exists():
        return [
            "Flasche",
            "Tasche",
            "Rucksack",
            "Kleidungsstück",
            "Sonstiges"
        ]

    labels = LABEL_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines()

    return [clean_label(x) for x in labels if x.strip()]

# ---------------------------------------------------------
# KI-MODELL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None

    try:
        import tensorflow as tf
        return tf.keras.models.load_model(
            MODEL_FILE,
            compile=False
        )
    except Exception:
        return None


def predict(image):
    model = load_model()

    if model is None:
        return None, None

    try:
        import numpy as np

        img = image.convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype="float32") / 255.0
        arr = np.expand_dims(arr, 0)

        prediction = model.predict(arr, verbose=0)[0]

        index = int(np.argmax(prediction))
        confidence = float(prediction[index])

        labels = load_labels()

        if index < len(labels):
            name = labels[index]
        else:
            name = "Unbekannt"

        return name, confidence

    except Exception:
        return None, None

# ---------------------------------------------------------
# BILD SPEICHERN
# ---------------------------------------------------------

def save_image(file):
    data = file.getvalue()

    image_id = hashlib.md5(data).hexdigest()
    path = IMAGES / f"{image_id}.jpg"

    image = Image.open(file).convert("RGB")
    image.save(path, "JPEG", quality=90)

    return path

# ---------------------------------------------------------
# SESSION
# ---------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected" not in st.session_state:
    st.session_state.selected = None

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

def header(back=False):

    left = ""
    right = ""

    if back:
        if st.button("←", key="back"):
            st.session_state.page = "start"
            st.rerun()

    profile = ASSETS / "profile.png"

    if profile.exists():
        right = (
            f'<img src="data:image/png;base64,'
            f'{__import__("base64").b64encode(profile.read_bytes()).decode()}'
            f'" class="profile-img">'
        )
    else:
        right = "◯"

    st.markdown(
        f"""
        <div class="app-header">
            <div>{left}</div>
            <div class="logo-title">FUNDBÜRO</div>
            <div style="text-align:right">{right}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

def footer():

    st.markdown("""
        <div class="footer">
            <div class="footer-tu-es">TU ES</div>
            <div class="footer-school">
                KATHARINEUM ZU LÜBECK
            </div>
            <div class="footer-symbol">◉</div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# BILDER-GRID
# ---------------------------------------------------------

def image_grid(rows):

    if not rows:
        st.info("Noch keine Fundstücke vorhanden.")
        return

    st.markdown('<div class="image-grid">', unsafe_allow_html=True)

    for row in rows[:9]:

        path = Path(row[1])

        if path.exists():
            st.image(
                str(path),
                width="stretch"
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Auswahl der Fundstücke unter dem Raster
    for row in rows[:9]:
        if st.button(
            row[2] if row[2] else "Fundstück",
            key=f"item_{row[0]}"
        ):
            st.session_state.selected = row[0]
            st.session_state.page = "detail"
            st.rerun()

# ---------------------------------------------------------
# STARTSEITE
# ---------------------------------------------------------

def start():

    header()

    st.markdown("""
        <div class="main-action">
            <span class="action-icon">↑</span>
            FOTO HOCHLADEN
        </div>
    """, unsafe_allow_html=True)

    if st.button("FOTO HOCHLADEN", key="upload_start"):
        st.session_state.page = "upload"
        st.rerun()

    st.markdown("""
        <div class="main-action">
            <span class="action-icon">⌕</span>
            KLEIDUNGSSTÜCK SUCHEN
        </div>
    """, unsafe_allow_html=True)

    if st.button(
        "KLEIDUNGSSTÜCK SUCHEN",
        key="search_start"
    ):
        st.session_state.page = "search"
        st.rerun()

    st.markdown(
        '<div class="section-title">LETZTE FUNDSTÜCKE</div>',
        unsafe_allow_html=True
    )

    image_grid(get_items())

    footer()

# ---------------------------------------------------------
# HOCHLADEN
# ---------------------------------------------------------

def upload():

    header(back=True)

    st.markdown(
        '<div class="section-title">FUNDSTÜCK HINZUFÜGEN</div>',
        unsafe_allow_html=True
    )

    camera = st.camera_input("Foto aufnehmen")

    uploaded = st.file_uploader(
        "Oder Foto auswählen",
        type=["jpg", "jpeg", "png"]
    )

    file = camera if camera is not None else uploaded

    if file is not None:

        image = Image.open(file).convert("RGB")

        st.image(
            image,
            caption="Ausgewähltes Fundstück",
            width="stretch"
        )

        # -------------------------------------------------
        # KI nur einmal pro Bild ausführen
        # -------------------------------------------------

        image_bytes = file.getvalue()
        image_id = hashlib.md5(image_bytes).hexdigest()

        if st.session_state.last_image != image_id:

            name, confidence = predict(image)

            st.session_state.ai_result = name
            st.session_state.ai_confidence = confidence
            st.session_state.last_image = image_id

        # -------------------------------------------------
        # KI-Ergebnis
        # -------------------------------------------------

        if st.session_state.ai_result:

            confidence = st.session_state.ai_confidence

            st.markdown(
                f"""
                <div class="ai-box">
                    <div class="ai-title">KI-ERKENNUNG</div>
                    <div>
                        Erkannt:
                        <b>{st.session_state.ai_result}</b>
                    </div>
                    <div>
                        Sicherheit:
                        <b>{confidence * 100:.0f} %</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        name = st.text_input(
            "Bezeichnung",
            value=st.session_state.ai_result or ""
        )

        farbe = st.text_input("Farbe")

        fundort = st.text_input(
            "Fundort",
            placeholder="z. B. Obere Turnhalle"
        )

        notizen = st.text_area(
            "Notizen",
            placeholder="Weitere Informationen zum Fundstück"
        )

        if st.button(
            "FUNDSTÜCK SPEICHERN",
            key="save_item"
        ):

            path = save_image(file)

            add_item(
                image_path=str(path),
                name=name or "Fundstück",
                kategorie=st.session_state.ai_result or "",
                farbe=farbe,
                gefunden_am="03.09.2026",
                fundort=fundort,
                notizen=notizen
            )

            st.session_state.page = "start"
            st.session_state.last_image = None
            st.session_state.ai_result = None
            st.session_state.ai_confidence = None

            st.rerun()

    footer()

# ---------------------------------------------------------
# SUCHE
# ---------------------------------------------------------

def search():

    header(back=True)

    query = st.text_input(
        "Suche",
        placeholder="(z. B.:) Flaschen",
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="section-title">'
        'ERGEBNISSE FÜR DIE SUCHE'
        '</div>',
        unsafe_allow_html=True
    )

    if query:
        results = get_items(query)
    else:
        results = get_items()

    image_grid(results)

    footer()

# ---------------------------------------------------------
# DETAILSEITE
# ---------------------------------------------------------

def detail():

    item = get_item(st.session_state.selected)

    if not item:
        st.session_state.page = "start"
        st.rerun()

    header(back=True)

    path = Path(item[1])

    if path.exists():

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                str(path),
                width="stretch"
            )

    st.markdown(
        f'<div class="section-title">'
        f'{item[2] or "FUNDSTÜCK"}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-box">

            <div class="info-label">GEFUNDEN AM:</div>
            <div class="info-text">
                {item[5] or "-"}
            </div>

            <div class="info-label">FUNDORT:</div>
            <div class="info-text">
                {item[6] or "-"}
            </div>

            <div class="info-label">NOTIZEN:</div>
            <div class="info-text">
                {item[7] or "-"}
            </div>

            <div class="info-label">AKTUELLER STANDORT:</div>
            <div>
                {item[8] or "Fundkiste"}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if item[9]:
        st.markdown(
            '<div class="assigned">'
            'FUNDSTÜCK ZUGEORDNET'
            '</div>',
            unsafe_allow_html=True
        )

    footer()

# ---------------------------------------------------------
# PROFIL
# ---------------------------------------------------------

def profile():

    header(back=True)

    st.markdown(
        '<div class="section-title">PROFIL</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class="info-box">
            <div class="info-label">KATHARINEUM ZU LÜBECK</div>
            <div>
                Fundbüro-App
            </div>
        </div>
    """, unsafe_allow_html=True)

    footer()

# ---------------------------------------------------------
# SEITENSTEUERUNG
# ---------------------------------------------------------

if st.session_state.page == "start":
    start()

elif st.session_state.page == "upload":
    upload()

elif st.session_state.page == "search":
    search()

elif st.session_state.page == "detail":
    detail()

elif st.session_state.page == "profile":
    profile()
