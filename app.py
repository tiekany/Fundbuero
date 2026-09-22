import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

BASE = Path(__file__).parent
ASSETS = BASE / "assets"
MODEL_DIR = BASE / "modell"
IMAGES = BASE / "bilder"
DATA = BASE / "daten"

ASSETS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
IMAGES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

DB = DATA / "fundbuero.db"
LOGO = ASSETS / "katharineum_logo.png"
PROFILE = ASSETS / "profile.png"
MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="FUNDBÜRO",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:#f4f4f4;
}

.block-container {
    max-width:950px;
    margin:auto;
    padding:20px 20px 130px;
}

* {
    font-family:Arial, Helvetica, sans-serif;
}

.main-title {
    text-align:center;
    font-size:38px;
    font-weight:900;
    margin:10px 0 20px;
}

.logo {
    text-align:center;
    margin-bottom:15px;
}

.logo img {
    max-width:200px;
    max-height:90px;
}

.profile {
    position:absolute;
    right:10px;
    top:10px;
}

.heading {
    font-size:22px;
    font-weight:900;
    margin:25px 0 12px;
}

.action {
    background:white;
    border:3px solid black;
    border-radius:4px;
    min-height:85px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:19px;
    font-weight:900;
    margin-bottom:10px;
}

div.stButton > button {
    background:white;
    color:black;
    border:2px solid black;
    border-radius:5px;
    font-weight:800;
    box-shadow:none;
}

div.stButton > button:hover {
    background:black;
    color:white;
    border-color:black;
}

div[data-baseweb="input"],
div[data-baseweb="textarea"] {
    background:white;
    border:2px solid black;
    border-radius:4px;
    box-shadow:none;
}

[data-testid="stImage"] img {
    border:3px solid black;
    border-radius:0;
}

.info {
    background:white;
    border:2px solid black;
    padding:13px;
    margin-bottom:8px;
}

.ai {
    background:white;
    border:3px solid black;
    padding:14px;
    text-align:center;
    font-weight:900;
    margin:12px 0;
}

.assigned {
    background:white;
    border:3px solid black;
    padding:15px;
    text-align:center;
    font-weight:900;
    margin-top:12px;
}

.footer {
    text-align:center;
    border-top:3px solid black;
    margin-top:30px;
    padding-top:20px;
}

.footer-tu {
    font-size:28px;
    font-weight:900;
}

.footer-school {
    font-size:14px;
    font-weight:900;
}

@media(max-width:700px) {
    .block-container {
        padding:12px 10px 120px;
    }

    .main-title {
        font-size:30px;
    }

    .action {
        min-height:75px;
        font-size:16px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected" not in st.session_state:
    st.session_state.selected = None

if "last_image" not in st.session_state:
    st.session_state.last_image = None

if "ai" not in st.session_state:
    st.session_state.ai = ("", 0)


def go(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# DATENBANK
# ============================================================

def db():
    return sqlite3.connect(DB)


def init_db():
    c = db()
    c.execute("""
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
    c.commit()
    c.close()


init_db()


def items(query=None):

    c = db()

    if query:
        q = f"%{query.lower()}%"
        rows = c.execute("""
            SELECT * FROM fundstuecke
            WHERE lower(name) LIKE ?
               OR lower(kategorie) LIKE ?
               OR lower(farbe) LIKE ?
               OR lower(fundort) LIKE ?
               OR lower(notizen) LIKE ?
            ORDER BY id DESC
        """, (q, q, q, q, q)).fetchall()

    else:
        rows = c.execute("""
            SELECT * FROM fundstuecke
            ORDER BY id DESC
        """).fetchall()

    c.close()
    return rows


def get_item(item_id):
    c = db()
    row = c.execute(
        "SELECT * FROM fundstuecke WHERE id=?",
        (item_id,)
    ).fetchone()
    c.close()
    return row


def add_item(data):
    c = db()
    c.execute("""
        INSERT INTO fundstuecke
        (image_path,name,kategorie,farbe,gefunden_am,
         fundort,notizen,aktueller_standort)
        VALUES (?,?,?,?,?,?,?,?)
    """, data)
    c.commit()
    c.close()


# ============================================================
# LABELS / KI
# ============================================================

def labels():
    if LABELS_PATH.exists():
        return [
            x.strip().split(maxsplit=1)[-1]
            for x in LABELS_PATH.read_text().splitlines()
            if x.strip()
        ]

    return [
        "Flasche",
        "Tasche",
        "Rucksack",
        "Kleidungsstück",
        "Sonstiges"
    ]


LABELS = labels()


@st.cache_resource
def model():
    if not MODEL_PATH.exists():
        return None

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
    except Exception:
        return None


def predict(image):

    m = model()

    if m is None:
        return "", 0

    try:
        image = image.convert("RGB").resize((224, 224))

        x = tf.convert_to_tensor(
            image,
            dtype=tf.float32
        )

        x = tf.expand_dims(
            x / 127.5 - 1,
            0
        )

        result = m(
            x,
            training=False
        ).numpy()[0]

        i = int(result.argmax())

        return (
            LABELS[i] if i < len(LABELS) else f"Klasse {i+1}",
            float(result[i])
        )

    except Exception:
        return "", 0


# ============================================================
# HEADER / FOOTER
# ============================================================

def header(back=False):

    if back:
        if st.button("←", key="back"):
            go("start")

    st.markdown(
        '<div class="main-title">FUNDBÜRO</div>',
        unsafe_allow_html=True
    )

    if PROFILE.exists():
        st.image(
            str(PROFILE),
            width=40
        )


def footer():

    st.markdown(
        """
        <div class="footer">
            <div class="footer-tu">TU ES</div>
            <div class="footer-school">
                KATHARINEUM ZU LÜBECK
            </div>
            ∞
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        if st.button("⌂", key="home"):
            go("start")

    with b:
        if st.button("⌕", key="search"):
            go("search")

    with c:
        if st.button("♙", key="profile"):
            go("profile")


# ============================================================
# FUNDSTÜCK-GITTER
# ============================================================

def grid(data):

    for i in range(0, len(data), 3):

        cols = st.columns(3)

        for col, item in zip(cols, data[i:i+3]):

            with col:

                if item[1] and Path(item[1]).exists():
                    st.image(
                        item[1],
                        width="stretch"
                    )

                st.markdown(
                    f"**{item[2] or 'Fundstück'}**"
                )

                if st.button(
                    "ANSEHEN",
                    key=f"view{item[0]}"
                ):
                    st.session_state.selected = item[0]
                    go("detail")


# ============================================================
# STARTSEITE
# ============================================================

def start():

    if LOGO.exists():

        st.markdown(
            '<div class="logo">',
            unsafe_allow_html=True
        )

        st.image(
            str(LOGO),
            width=200
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    header()


    st.markdown(
        '<div class="action">↑ &nbsp; FOTO HOCHLADEN</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "FOTO HOCHLADEN",
        key="upload"
    ):
        go("upload")


    st.markdown(
        '<div class="action">⌕ &nbsp; KLEIDUNGSSTÜCK SUCHEN</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "KLEIDUNGSSTÜCK SUCHEN",
        key="search_button"
    ):
        go("search")


    st.markdown(
        '<div class="heading">LETZTE FUNDSTÜCKE</div>',
        unsafe_allow_html=True
    )

    grid(items()[:6])

    footer()


# ============================================================
# UPLOAD
# ============================================================

def upload():

    header(True)

    st.markdown(
        '<div class="heading">FOTO HOCHLADEN</div>',
        unsafe_allow_html=True
    )

    camera = st.camera_input("Foto")

    uploaded = st.file_uploader(
        "Oder Bild auswählen",
        type=["jpg", "jpeg", "png", "webp"]
    )

    file = camera or uploaded

    if not file:
        footer()
        return

    raw = file.getvalue()
    image_id = hashlib.md5(raw).hexdigest()
    image = Image.open(file).convert("RGB")

    st.image(
        image,
        width="stretch"
    )


    # --------------------------------------------------------
    # KI nur einmal pro Bild
    # --------------------------------------------------------

    if st.session_state.last_image != image_id:

        st.session_state.last_image = image_id

        with st.spinner("Bild wird erkannt..."):
            st.session_state.ai = predict(image)


    label, confidence = st.session_state.ai


    if label:

        st.markdown(
            f"""
            <div class="ai">
                Erkannt: {label}<br>
                Sicherheit: {confidence * 100:.0f} %
            </div>
            """,
            unsafe_allow_html=True
        )


    name = st.text_input(
        "Bezeichnung",
        value=label
    )

    category = st.text_input(
        "Kategorie",
        value=label
    )

    color = st.text_input(
        "Farbe"
    )

    date = st.date_input(
        "Gefunden am"
    )

    place = st.text_input(
        "Fundort"
    )

    notes = st.text_area(
        "Notizen"
    )

    location = st.text_input(
        "Aktueller Standort",
        value="Fundkiste"
    )


    if st.button(
        "FUNDSTÜCK SPEICHERN"
    ):

        filename = (
            uuid.uuid4().hex
            + ".jpg"
        )

        path = IMAGES / filename

        image.save(
            path,
            quality=90
        )

        add_item(
            (
                str(path),
                name,
                category,
                color,
                str(date),
                place,
                notes,
                location
            )
        )

        st.session_state.last_image = None

        go("start")


    footer()


# ============================================================
# SUCHE
# ============================================================

def search():

    header(True)

    st.markdown(
        '<div class="heading">⌕ SUCHEN</div>',
        unsafe_allow_html=True
    )

    query = st.text_input(
        "Suche",
        placeholder="(z.B.:) Flaschen",
        label_visibility="collapsed"
    )


    if query:

        st.markdown(
            f"""
            <div class="heading">
                ERGEBNISSE FÜR „{query.upper()}“
            </div>
            """,
            unsafe_allow_html=True
        )

        grid(items(query))

    else:

        st.markdown(
            '<div class="heading">ERGEBNISSE</div>',
            unsafe_allow_html=True
        )

        grid(items())


    footer()


# ============================================================
# DETAIL
# ============================================================

def detail():

    item = get_item(
        st.session_state.selected
    )

    if not item:
        go("start")
        return


    header(True)


    st.markdown(
        '<div class="heading">FUNDBÜRO</div>',
        unsafe_allow_html=True
    )


    if item[1] and Path(item[1]).exists():

        st.image(
            item[1],
            width="stretch"
        )


    st.markdown(
        f'<div class="heading">{item[2]}</div>',
        unsafe_allow_html=True
    )


    details = [
        ("GEFUNDEN AM:", item[5]),
        ("FUNDORT:", item[6]),
        ("NOTIZEN:", item[7]),
        ("AKTUELLER STANDORT:", item[8]),
    ]


    for title, value in details:

        st.markdown(
            f"""
            <div class="info">
                <b>{title}</b><br>
                {value or "Keine Angabe"}
            </div>
            """,
            unsafe_allow_html=True
        )


    if item[9]:

        st.markdown(
            '<div class="assigned">FUNDSTÜCK ZUGEORDNET</div>',
            unsafe_allow_html=True
        )


    else:

        if st.button(
            "FUNDSTÜCK ZUORDNEN"
        ):

            c = db()

            c.execute(
                "UPDATE fundstuecke SET zugeordnet=1 WHERE id=?",
                (item[0],)
            )

            c.commit()
            c.close()

            st.rerun()


    footer()


# ============================================================
# PROFIL
# ============================================================

def profile():

    header(True)

    st.markdown(
        '<div class="heading">PROFIL</div>',
        unsafe_allow_html=True
    )


    if PROFILE.exists():
        st.image(
            str(PROFILE),
            width=80
        )


    if LOGO.exists():
        st.image(
            str(LOGO),
            width=200
        )


    st.markdown(
        """
        <div class="info">
            <b>KATHARINEUM ZU LÜBECK</b><br><br>
            Digitales Fundbüro der Schule.
        </div>

        <div class="info">
            <b>FUNKTIONEN</b><br><br>
            • Foto aufnehmen<br>
            • Foto hochladen<br>
            • KI-Erkennung<br>
            • Fundstücke suchen<br>
            • Fundstücke zuordnen
        </div>
        """,
        unsafe_allow_html=True
    )


    footer()


# ============================================================
# ROUTER
# ============================================================

pages = {
    "start": start,
    "upload": upload,
    "search": search,
    "detail": detail,
    "profile": profile
}

pages.get(
    st.session_state.page,
    start
)()
