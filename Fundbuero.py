import sqlite3
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# KATH-LOST AND FOUND
# ============================================================

BASE = Path(__file__).resolve().parent

MODEL_PATH = BASE / "modell" / "keras_model.h5"
LABELS_PATH = BASE / "modell" / "labels.txt"

IMAGE_DIR = BASE / "bilder"
DATA_DIR = BASE / "daten"
DB_PATH = DATA_DIR / "fundbuero.db"

IMAGE_SIZE = (224, 224)


# ============================================================
# BILD FÜR DAS H5-MODELL VORBEREITEN
# ============================================================

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    pixels = tf.keras.utils.img_to_array(image)

    # Typisches Preprocessing des Teachable-Machine-Keras-Exports
    pixels = (pixels / 127.5) - 1.0

    return tf.expand_dims(pixels, 0)


# ============================================================
# H5-MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"H5-Modell nicht gefunden:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# KLASSENNAMEN LADEN
# ============================================================

def load_labels():

    if LABELS_PATH.exists():

        labels = [
            line.strip()
            for line in LABELS_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if labels:
            return labels

    # Platzhalter
    return [
        "Klasse 1",
        "Klasse 2",
        "Klasse 3",
        "Klasse 4",
        "Klasse 5",
    ]


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify(image):

    model = load_model()
    labels = load_labels()

    image_tensor = prepare_image(image)

    prediction = model.predict(
        image_tensor,
        verbose=0
    )[0]

    if len(prediction) != len(labels):

        raise ValueError(
            f"Das Modell liefert "
            f"{len(prediction)} Klassen, "
            f"aber labels.txt enthält "
            f"{len(labels)} Klassen."
        )

    best_index = int(prediction.argmax())

    ranking = sorted(
        [
            (
                labels[i],
                float(prediction[i])
            )
            for i in range(len(labels))
        ],
        key=lambda x: x[1],
        reverse=True
    )

    return (
        labels[best_index],
        float(prediction[best_index]),
        ranking
    )


# ============================================================
# DATENBANK
# ============================================================

def connect():

    DATA_DIR.mkdir(
        exist_ok=True
    )

    IMAGE_DIR.mkdir(
        exist_ok=True
    )

    connection = sqlite3.connect(
        DB_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    with connect() as con:

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fundstuecke (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                bild TEXT NOT NULL,

                kategorie TEXT NOT NULL,

                bezeichnung TEXT,

                funddatum TEXT,

                fundort TEXT,

                notizen TEXT,

                standort TEXT,

                farbe TEXT,

                status TEXT DEFAULT 'Gefunden',

                ki_kategorie TEXT,

                ki_sicherheit REAL
            )
            """
        )

        con.commit()


# ============================================================
# FUNDSTÜCK SPEICHERN
# ============================================================

def save_item(
    image,
    category,
    title,
    found_date,
    location,
    notes,
    current_location,
    color,
    ki_category,
    confidence
):

    filename = (
        f"{uuid.uuid4().hex}.jpg"
    )

    image_path = (
        IMAGE_DIR / filename
    )

    image.convert("RGB").save(
        image_path,
        "JPEG",
        quality=90
    )

    with connect() as con:

        cursor = con.execute(
            """
            INSERT INTO fundstuecke
            (
                bild,
                kategorie,
                bezeichnung,
                funddatum,
                fundort,
                notizen,
                standort,
                farbe,
                status,
                ki_kategorie,
                ki_sicherheit
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (
                str(image_path),
                category,
                title,
                found_date,
                location,
                notes,
                current_location,
                color,
                "Gefunden",
                ki_category,
                confidence
            )
        )

        con.commit()

        return cursor.lastrowid


# ============================================================
# FUNDSTÜCK ABRUFEN
# ============================================================

def get_item(item_id):

    with connect() as con:

        return con.execute(
            """
            SELECT *
            FROM fundstuecke
            WHERE id = ?
            """,
            (item_id,)
        ).fetchone()


# ============================================================
# LETZTE FUNDSTÜCKE
# ============================================================

def get_latest(limit=6):

    with connect() as con:

        return con.execute(
            """
            SELECT *
            FROM fundstuecke
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()


# ============================================================
# FUNDSTÜCKE SUCHEN
# ============================================================

def search_items(
    text="",
    category="Alle",
    color="Alle"
):

    sql = """
        SELECT *
        FROM fundstuecke
        WHERE
        (
            bezeichnung LIKE ?
            OR kategorie LIKE ?
            OR fundort LIKE ?
            OR notizen LIKE ?
            OR farbe LIKE ?
        )
    """

    value = f"%{text}%"

    params = [
        value,
        value,
        value,
        value,
        value
    ]

    if category != "Alle":

        sql += """
            AND kategorie = ?
        """

        params.append(category)

    if color != "Alle":

        sql += """
            AND farbe = ?
        """

        params.append(color)

    sql += """
        ORDER BY id DESC
    """

    with connect() as con:

        return con.execute(
            sql,
            params
        ).fetchall()


# ============================================================
# STATUS ÄNDERN
# ============================================================

def set_status(
    item_id,
    status
):

    with connect() as con:

        con.execute(
            """
            UPDATE fundstuecke
            SET status = ?
            WHERE id = ?
            """,
            (
                status,
                item_id
            )
        )

        con.commit()


# ============================================================
# STREAMLIT EINSTELLUNGEN
# ============================================================

st.set_page_config(

    page_title="KATH-LOST AND FOUND",

    page_icon="🧥",

    layout="centered"
)


init_db()


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: white;
    }

    .block-container {

        max-width: 560px;

        padding-top: 1rem;

        padding-bottom: 2rem;
    }


    .phone {

        border: 4px solid #222;

        border-radius: 42px;

        padding: 20px 17px;

        background: white;
    }


    .logo {

        text-align: center;

        font-size: 35px;

        font-weight: 950;

        letter-spacing: -2px;

        margin-bottom: 15px;
    }


    .section {

        font-size: 23px;

        font-weight: 950;

        margin: 17px 0 8px;

        letter-spacing: -1px;
    }


    .result {

        font-size: 20px;

        font-weight: 950;

        margin: 8px 0;
    }


    .detail-box {

        border: 3px solid #222;

        margin-top: 10px;
    }


    .detail-row {

        border-bottom: 2px solid #222;

        padding: 7px;

        font-size: 15px;
    }


    .detail-row:last-child {

        border-bottom: none;
    }


    .assigned {

        border: 3px solid green;

        color: green;

        font-weight: 950;

        text-align: center;

        padding: 8px;

        margin-top: 8px;
    }


    .brand-bottom {

        text-align: center;

        font-weight: 900;

        font-size: 12px;

        margin-top: 18px;
    }


    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {

        border: 3px solid #222;

        border-radius: 4px;

        background: white;

        color: #222;

        font-weight: 900;

        min-height: 45px;
    }


    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {

        border-color: #222;

        color: #222;
    }


    [data-testid="stFileUploader"] {

        border: 3px solid #222;

        border-radius: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

def top():

    st.markdown(
        """
        <div class="phone">

        <div class="logo">
            FUNDBÜRO
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

def bottom():

    st.markdown(
        """
        <div class="brand-bottom">

            <span
                style="
                    color:#d00;
                    font-size:24px;
                "
            >
                TU<br>ES
            </span>

            KATHARINEUM ZU LÜBECK ⚓

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SEITENWECHSEL
# ============================================================

def go_to(page_name):

    st.session_state["page"] = page_name

    st.rerun()


# ============================================================
# STANDARDSEITE
# ============================================================

if "page" not in st.session_state:

    st.session_state["page"] = "start"


# ============================================================
# STARTSEITE
# ============================================================

def start_page():

    top()

    st.markdown(
        """
        <div
            style="
                text-align:right;
                font-size:25px;
            "
        >
            ●
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FOTO HOCHLADEN
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section">
            FOTO HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "📷 FOTO AUFNEHMEN / DATEI AUSWÄHLEN",
        use_container_width=True
    ):

        go_to("upload")


    # --------------------------------------------------------
    # SUCHEN
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section">
            KLEIDUNGSSTÜCK SUCHEN
        </div>
        """,
        unsafe_allow_html=True
    )


    search = st.text_input(

        "Suche",

        placeholder="z.B. Flasche",

        label_visibility="collapsed"
    )


    if st.button(
        "🔍 SUCHEN",
        use_container_width=True
    ):

        st.session_state["search_text"] = search

        go_to("search")


    # --------------------------------------------------------
    # LETZTE FUNDSTÜCKE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section">
            LETZTE FUNDSTÜCKE
        </div>
        """,
        unsafe_allow_html=True
    )


    items = get_latest()


    if not items:

        st.caption(
            "Noch keine Fundstücke vorhanden."
        )


    else:

        for start in range(
            0,
            len(items),
            3
        ):

            cols = st.columns(3)


            for col, item in zip(
                cols,
                items[start:start + 3]
            ):

                with col:

                    st.image(
                        item["bild"],
                        use_container_width=True
                    )


                    label = (
                        item["bezeichnung"]
                        or
                        item["kategorie"]
                    )


                    if st.button(

                        label,

                        key=f"latest_{item['id']}",

                        use_container_width=True
                    ):

                        st.session_state[
                            "selected_id"
                        ] = item["id"]

                        go_to("detail")


    st.markdown(
        """
        <div
            style="
                font-size:12px;
                margin-top:10px;
            "
        >
            Optional: Suche kann nach Farbe gefiltert werden.
        </div>
        """,
        unsafe_allow_html=True
    )


    bottom()


# ============================================================
# FOTO AUFNEHMEN / HOCHLADEN
# ============================================================

def upload_page():

    top()


    if st.button(
        "← ZURÜCK",
        use_container_width=True
    ):

        go_to("start")


    st.markdown(
        """
        <div class="section">
            FOTO HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True
    )


    # Kamera

    camera = st.camera_input(
        "Foto aufnehmen"
    )


    # Datei

    uploaded = st.file_uploader(

        "oder Datei auswählen",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    source = (
        camera
        if camera is not None
        else uploaded
    )


    if source is None:

        st.info(
            "Nimm ein Foto auf oder lade ein Bild hoch."
        )

        bottom()

        return


    image = Image.open(source)


    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------------
    # KI
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Gegenstand wird erkannt ..."
        ):

            predicted, confidence, ranking = classify(
                image
            )

    except Exception as exc:

        st.error(
            f"KI-Erkennung fehlgeschlagen:\n{exc}"
        )

        bottom()

        return


    st.markdown(
        f"""
        <div class="result">
            ERKANNT: {predicted}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.progress(

        min(
            max(
                confidence,
                0
            ),
            1
        ),

        text=f"Sicherheit: {confidence:.1%}"
    )


    # Weitere Klassen

    with st.expander(
        "Weitere Erkennungen"
    ):

        for label, probability in ranking:

            st.write(
                f"{label}: {probability:.1%}"
            )


    # --------------------------------------------------------
    # FUNDSTÜCK SPEICHERN
    # --------------------------------------------------------

    st.markdown(
        "### Fundstück speichern"
    )


    labels = load_labels()


    with st.form(
        "new_item"
    ):


        category = st.selectbox(

            "Kategorie",

            labels,

            index=(
                labels.index(predicted)
                if predicted in labels
                else 0
            )
        )


        title = st.text_input(

            "Bezeichnung",

            placeholder="z.B. blaue Trinkflasche"
        )


        found_date = st.date_input(

            "Gefunden am",

            date.today()
        )


        location = st.text_input(

            "Fundort",

            placeholder="z.B. Obere Turnhalle"
        )


        notes = st.text_area(

            "Notizen",

            placeholder=(
                "z.B. lag hinter dem Mattenwagen"
            )
        )


        current_location = st.text_input(

            "Aktueller Standort",

            placeholder="z.B. Fundkiste"
        )


        color = st.text_input(

            "Farbe",

            placeholder="z.B. blau"
        )


        save = st.form_submit_button(

            "FUNDSTÜCK SPEICHERN",

            use_container_width=True
        )


    if save:

        item_id = save_item(

            image,

            category,

            title,

            found_date.isoformat(),

            location,

            notes,

            current_location,

            color,

            predicted,

            confidence
        )


        st.session_state[
            "selected_id"
        ] = item_id


        st.success(
            "Fundstück wurde gespeichert."
        )


    bottom()


# ============================================================
# SUCHSEITE
# ============================================================

def search_page():

    top()


    if st.button(
        "← ZUR STARTSEITE",
        use_container_width=True
    ):

        go_to("start")


    st.markdown(
        """
        <div class="section">
            KLEIDUNGSSTÜCK SUCHEN
        </div>
        """,
        unsafe_allow_html=True
    )


    text = st.text_input(

        "Suchbegriff",

        value=st.session_state.get(
            "search_text",
            ""
        ),

        placeholder="z.B. Flaschen"
    )


    # Alle Fundstücke

    all_items = get_latest(
        10000
    )


    categories = ["Alle"] + sorted(
        {
            item["kategorie"]
            for item in all_items
        }
    )


    colors = ["Alle"] + sorted(
        {
            item["farbe"]
            for item in all_items
            if item["farbe"]
        }
    )


    c1, c2 = st.columns(2)


    with c1:

        category = st.selectbox(
            "Kategorie",
            categories
        )


    with c2:

        color = st.selectbox(
            "Farbe",
            colors
        )


    results = search_items(

        text,

        category,

        color
    )


    st.markdown(
        f"""
        <div class="result">
            ERGEBNISSE FÜR „{text or "ALLE"}“
        </div>
        """,
        unsafe_allow_html=True
    )


    if not results:

        st.info(
            "Keine Fundstücke gefunden."
        )


    else:

        for start in range(
            0,
            len(results),
            3
        ):

            cols = st.columns(3)


            for col, item in zip(
                cols,
                results[start:start + 3]
            ):

                with col:

                    st.image(
                        item["bild"],
                        use_container_width=True
                    )


                    if st.button(

                        "Details",

                        key=f"result_{item['id']}",

                        use_container_width=True
                    ):

                        st.session_state[
                            "selected_id"
                        ] = item["id"]

                        go_to("detail")


    st.divider()


    if st.button(
        "📷 MIT FOTO SUCHEN",
        use_container_width=True
    ):

        go_to("photo_search")


    bottom()


# ============================================================
# FOTO-SUCHE
# ============================================================

def photo_search():

    top()


    if st.button(
        "← ZURÜCK",
        use_container_width=True
    ):

        go_to("search")


    st.markdown(
        """
        <div class="section">
            MIT FOTO SUCHEN
        </div>
        """,
        unsafe_allow_html=True
    )


    st.caption(
        """
        Fotografiere einen Gegenstand oder lade ein Bild hoch.
        Das H5-Modell erkennt anschließend die Kategorie und
        zeigt gespeicherte Fundstücke derselben Kategorie.
        """
    )


    camera = st.camera_input(
        "Suchfoto aufnehmen"
    )


    uploaded = st.file_uploader(

        "oder Suchfoto auswählen",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    source = (
        camera
        if camera is not None
        else uploaded
    )


    if source is None:

        bottom()

        return


    image = Image.open(source)


    st.image(
        image,
        use_container_width=True
    )


    try:

        predicted, confidence, _ = classify(
            image
        )

    except Exception as exc:

        st.error(
            f"KI-Erkennung fehlgeschlagen:\n{exc}"
        )

        bottom()

        return


    st.markdown(

        f"""
        <div class="assigned">
            ERKANNT: {predicted}
            –
            {confidence:.1%}
        </div>
        """,

        unsafe_allow_html=True
    )


    # Nur die Kategorie als Suchfilter

    results = search_items(
        category=predicted
    )


    st.markdown(
        """
        <div class="result">
            PASSENDE FUNDSTÜCKE
        </div>
        """,
        unsafe_allow_html=True
    )


    if not results:

        st.info(
            "Keine passenden Fundstücke gespeichert."
        )


    else:

        for start in range(
            0,
            len(results),
            3
        ):

            cols = st.columns(3)


            for col, item in zip(
                cols,
                results[start:start + 3]
            ):

                with col:

                    st.image(
                        item["bild"],
                        use_container_width=True
                    )


                    if st.button(

                        "Details",

                        key=f"photo_{item['id']}",

                        use_container_width=True
                    ):

                        st.session_state[
                            "selected_id"
                        ] = item["id"]

                        go_to("detail")


    bottom()


# ============================================================
# DETAILSEITE
# ============================================================

def detail_page():

    item = get_item(

        st.session_state.get(
            "selected_id"
        )
    )


    if not item:

        go_to("start")


    top()


    if st.button(
        "← ZURÜCK",
        use_container_width=True
    ):

        go_to("search")


    st.markdown(

        f"""
        <div class="result">
            {item["kategorie"].upper()}
        </div>
        """,

        unsafe_allow_html=True
    )


    st.image(
        item["bild"],
        use_container_width=True
    )


    st.markdown(

        f"""
        <div class="detail-box">

            <div class="detail-row">
                <b>GEFUNDEN AM:</b>
                {item["funddatum"] or "-"}
            </div>

            <div class="detail-row">
                <b>FUNDORT:</b>
                {item["fundort"] or "-"}
            </div>

            <div class="detail-row">
                <b>NOTIZEN:</b>
                {item["notizen"] or "-"}
            </div>

            <div class="detail-row">
                <b>AKTUELLER STANDORT:</b>
                {item["standort"] or "-"}
            </div>

            <div class="detail-row">
                <b>FARBE:</b>
                {item["farbe"] or "-"}
            </div>

        </div>
        """,

        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FUNDSTÜCK ZUORDNEN
    # --------------------------------------------------------

    if item["status"] == "Gefunden":

        if st.button(

            "FUNDSTÜCK ZUGEORDNET",

            use_container_width=True
        ):

            set_status(

                item["id"],

                "Zugeordnet"
            )

            st.rerun()


    else:

        st.markdown(

            """
            <div class="assigned">
                FUNDSTÜCK ZUGEORDNET
            </div>
            """,

            unsafe_allow_html=True
        )


    bottom()


# ============================================================
# ROUTER
# ============================================================

current_page = st.session_state["page"]


if current_page == "start":

    start_page()


elif current_page == "upload":

    upload_page()


elif current_page == "search":

    search_page()


elif current_page == "photo_search":

    photo_search()


elif current_page == "detail":

    detail_page()


else:

    start_page()
