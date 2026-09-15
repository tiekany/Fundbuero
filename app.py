import sqlite3
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# EINSTELLUNGEN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "modell" / "keras_model.h5"
LABELS_PATH = BASE_DIR / "modell" / "labels.txt"

IMAGE_DIR = BASE_DIR / "bilder"
DATA_DIR = BASE_DIR / "daten"
DATABASE = DATA_DIR / "fundbuero.db"

IMAGE_SIZE = (224, 224)


# ============================================================
# H5-KOMPATIBILITÄT
#
# Dein Teachable-Machine-Modell wurde mit Keras 2.4.0
# gespeichert.
#
# Beim Laden neuer Keras-Versionen kommt teilweise:
#
# Unrecognized keyword arguments passed to DepthwiseConv2D:
# {'groups': 1}
#
# Deshalb akzeptieren wir "groups" und entfernen es,
# bevor der normale Layer geladen wird.
# ============================================================

class CompatibleDepthwiseConv2D(
    tf.keras.layers.DepthwiseConv2D
):

    def __init__(
        self,
        *args,
        groups=None,
        **kwargs
    ):

        # Das alte Modell speichert groups=1.
        # DepthwiseConv2D benötigt dieses Argument hier nicht.

        super().__init__(
            *args,
            **kwargs
        )


# ============================================================
# MODELL LADEN
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Das H5-Modell wurde nicht gefunden.\n\n"
            f"Gesucht wurde:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "DepthwiseConv2D":
                CompatibleDepthwiseConv2D
        }
    )

    return model


# ============================================================
# KLASSENNAMEN
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

    # Dein Modell besitzt 5 Ausgabeklassen.
    return [
        "Klasse 1",
        "Klasse 2",
        "Klasse 3",
        "Klasse 4",
        "Klasse 5"
    ]


# ============================================================
# BILD VORBEREITEN
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # Teachable Machine Keras Export
    image_array = (
        image_array / 127.5
    ) - 1.0

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# KI-KLASSIFIZIERUNG
# ============================================================

def classify_image(image):

    model = load_model()

    labels = load_labels()

    prepared = prepare_image(
        image
    )

    prediction = model.predict(
        prepared,
        verbose=0
    )[0]

    if len(prediction) != len(labels):

        raise ValueError(
            "Die Anzahl der Klassen passt nicht.\n"
            f"Modell: {len(prediction)}\n"
            f"labels.txt: {len(labels)}"
        )

    best_index = int(
        prediction.argmax()
    )

    best_label = labels[
        best_index
    ]

    confidence = float(
        prediction[best_index]
    )

    ranking = sorted(
        [
            (
                labels[i],
                float(prediction[i])
            )
            for i in range(
                len(labels)
            )
        ],
        key=lambda x: x[1],
        reverse=True
    )

    return (
        best_label,
        confidence,
        ranking
    )


# ============================================================
# DATENBANK
# ============================================================

def get_connection():

    DATA_DIR.mkdir(
        exist_ok=True
    )

    IMAGE_DIR.mkdir(
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def init_database():

    with get_connection() as con:

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

def save_fundstueck(
    image,
    category,
    name,
    found_date,
    found_location,
    notes,
    current_location,
    color,
    ai_category,
    confidence
):

    filename = (
        uuid.uuid4().hex
        + ".jpg"
    )

    image_path = (
        IMAGE_DIR / filename
    )

    image.convert(
        "RGB"
    ).save(
        image_path,
        "JPEG",
        quality=90
    )

    with get_connection() as con:

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
                name,
                found_date,
                found_location,
                notes,
                current_location,
                color,
                "Gefunden",
                ai_category,
                confidence
            )
        )

        con.commit()

        return cursor.lastrowid


# ============================================================
# FUNDSTÜCK HOLEN
# ============================================================

def get_fundstueck(
    item_id
):

    with get_connection() as con:

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

def get_latest(
    limit=6
):

    with get_connection() as con:

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
# SUCHE
# ============================================================

def search_fundstuecke(
    search_text="",
    category="Alle",
    color="Alle"
):

    text = f"%{search_text}%"

    query = """
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

    parameters = [
        text,
        text,
        text,
        text,
        text
    ]

    if category != "Alle":

        query += """
            AND kategorie = ?
        """

        parameters.append(
            category
        )

    if color != "Alle":

        query += """
            AND farbe = ?
        """

        parameters.append(
            color
        )

    query += """
        ORDER BY id DESC
    """

    with get_connection() as con:

        return con.execute(
            query,
            parameters
        ).fetchall()


# ============================================================
# STATUS
# ============================================================

def update_status(
    item_id,
    status
):

    with get_connection() as con:

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
# STREAMLIT
# ============================================================

st.set_page_config(

    page_title="KATH-LOST AND FOUND",

    page_icon="🧥",

    layout="centered"
)


init_database()


# ============================================================
# CSS
#
# Stark an deiner Zeichnung orientiert:
#
# - Smartphone-Rahmen
# - runde Ecken
# - dicke schwarze Linien
# - handschriftlicher Look
# - 3x2 bzw. 3x4 Bilder
# - TU ES unten
# - grüner Zuordnungsbutton
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       SEITE
       ------------------------------------------------------- */

    .stApp {

        background: #ffffff;
    }


    .block-container {

        max-width: 430px;

        padding-top: 15px;

        padding-left: 8px;

        padding-right: 8px;

        padding-bottom: 15px;
    }


    /* -------------------------------------------------------
       HANDY
       ------------------------------------------------------- */

    .phone {

        width: 100%;

        min-height: 780px;

        border:
            4px
            solid
            #111111;

        border-radius:
            58px;

        padding:
            30px
            22px
            22px
            22px;

        box-sizing:
            border-box;

        background:
            white;
    }


    /* -------------------------------------------------------
       LOGO
       ------------------------------------------------------- */

    .header {

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        position:
            relative;

        margin-bottom:
            30px;
    }


    .logo {

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-size:
            34px;

        font-weight:
            900;

        letter-spacing:
            -2px;

        color:
            #111111;

        text-align:
            center;
    }


    .profile {

        position:
            absolute;

        right:
            1px;

        top:
            2px;

        font-size:
            23px;

        font-family:
            Arial,
            sans-serif;
    }


    /* -------------------------------------------------------
       HAUPTBUTTONS
       ------------------------------------------------------- */

    .big-button {

        height:
            49px;

        width:
            100%;

        border:
            4px
            solid
            #111111;

        box-sizing:
            border-box;

        display:
            flex;

        align-items:
            center;

        justify-content:
            space-between;

        padding:
            0
            10px;

        margin-bottom:
            27px;

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-size:
            20px;

        font-weight:
            900;

        letter-spacing:
            -1px;
    }


    .big-button-icon {

        font-family:
            Arial,
            sans-serif;

        font-size:
            23px;

        font-weight:
            normal;
    }


    /* -------------------------------------------------------
       ÜBERSCHRIFTEN
       ------------------------------------------------------- */

    .section-title {

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-size:
            22px;

        font-weight:
            900;

        letter-spacing:
            -1px;

        margin:
            7px
            4px
            11px
            4px;

        line-height:
            1;
    }


    /* -------------------------------------------------------
       BILD-GRID
       ------------------------------------------------------- */

    .grid {

        display:
            grid;

        grid-template-columns:
            repeat(3, 1fr);

        gap:
            8px;

        width:
            100%;
    }


    .grid-image {

        width:
            100%;

        aspect-ratio:
            1 / 1;

        object-fit:
            contain;

        border:
            4px
            solid
            #111111;

        box-sizing:
            border-box;

        background:
            white;

        padding:
            2px;
    }


    /* -------------------------------------------------------
       SUCHFELD
       ------------------------------------------------------- */

    .search-field {

        width:
            100%;

        height:
            45px;

        border:
            4px
            solid
            #111111;

        box-sizing:
            border-box;

        display:
            flex;

        align-items:
            center;

        justify-content:
            space-between;

        padding:
            0
            9px;

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-size:
            18px;

        font-weight:
            900;

        margin-bottom:
            10px;
    }


    .search-icon {

        font-size:
            25px;

        font-family:
            Arial,
            sans-serif;
    }


    /* -------------------------------------------------------
       DETAIL-BILD
       ------------------------------------------------------- */

    .detail-image {

        width:
            185px;

        height:
            185px;

        margin:
            8px
            auto
            15px
            auto;

        border:
            10px
            solid
            #111111;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        box-sizing:
            border-box;

        background:
            white;

        overflow:
            hidden;
    }


    .detail-image img {

        width:
            100%;

        height:
            100%;

        object-fit:
            contain;
    }


    /* -------------------------------------------------------
       DETAIL-INFORMATIONEN
       ------------------------------------------------------- */

    .details {

        border:
            4px
            solid
            #111111;

        width:
            100%;

        box-sizing:
            border-box;
    }


    .detail-row {

        min-height:
            27px;

        border-bottom:
            3px
            solid
            #111111;

        padding:
            3px
            5px;

        box-sizing:
            border-box;

        font-family:
            "Comic Neue",
            Arial,
            sans-serif;

        font-size:
            14px;

        font-weight:
            700;

        line-height:
            1.1;
    }


    .detail-row:last-child {

        border-bottom:
            none;
    }


    .detail-label {

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-weight:
            900;

        font-size:
            15px;
    }


    /* -------------------------------------------------------
       GRÜNER BUTTON
       ------------------------------------------------------- */

    .assigned {

        border:
            4px
            solid
            #07952c;

        color:
            #07952c;

        text-align:
            center;

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-size:
            17px;

        font-weight:
            900;

        padding:
            3px;

        box-sizing:
            border-box;
    }


    /* -------------------------------------------------------
       FOOTER
       ------------------------------------------------------- */

    .footer {

        display:
            flex;

        justify-content:
            center;

        align-items:
            center;

        gap:
            8px;

        margin-top:
            18px;
    }


    .tu-es {

        color:
            #e00000;

        font-family:
            Arial,
            sans-serif;

        font-size:
            34px;

        font-weight:
            300;

        line-height:
            .82;

        padding-right:
            8px;

        border-right:
            3px
            solid
            #111111;
    }


    .kath {

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif;

        font-size:
            11px;

        line-height:
            1.0;

        font-weight:
            900;
    }


    .wheel {

        font-size:
            42px;

        line-height:
            1;
    }


    /* -------------------------------------------------------
       STREAMLIT BUTTONS
       ------------------------------------------------------- */

    div.stButton > button {

        border:
            4px
            solid
            #111111 !important;

        border-radius:
            0 !important;

        background:
            white !important;

        color:
            #111111 !important;

        font-family:
            "Comic Neue",
            "Arial Black",
            sans-serif !important;

        font-weight:
            900 !important;

        font-size:
            16px !important;

        min-height:
            42px;

        box-shadow:
            none !important;
    }


    div.stButton > button:hover {

        background:
            #f4f4f4 !important;

        border-color:
            #111111 !important;
    }


    /* -------------------------------------------------------
       STREAMLIT INPUT
       ------------------------------------------------------- */

    div[data-baseweb="input"] {

        border:
            4px
            solid
            #111111 !important;

        border-radius:
            0 !important;
    }


    div[data-baseweb="input"] input {

        font-family:
            "Comic Neue",
            Arial,
            sans-serif !important;

        font-weight:
            700 !important;
    }


    /* -------------------------------------------------------
       UPLOADER
       ------------------------------------------------------- */

    [data-testid="stFileUploader"] {

        border:
            4px
            solid
            #111111;

        border-radius:
            0;

        padding:
            5px;
    }


    /* -------------------------------------------------------
       KAMERA
       ------------------------------------------------------- */

    [data-testid="stCameraInput"] {

        border:
            4px
            solid
            #111111;

        border-radius:
            0;
    }


    /* -------------------------------------------------------
       MOBILE
       ------------------------------------------------------- */

    @media (max-width: 480px) {

        .block-container {

            padding:
                4px;
        }

        .phone {

            border-radius:
                45px;

            padding:
                25px
                17px
                18px
                17px;
        }

        .logo {

            font-size:
                30px;
        }

        .big-button {

            font-size:
                18px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

def show_header():

    st.markdown(
        """
        <div class="phone">

            <div class="header">

                <div class="logo">
                    FUNDBÜRO
                </div>

                <div class="profile">
                    ●
                </div>

            </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

def show_footer():

    st.markdown(
        """
            <div class="footer">

                <div class="tu-es">
                    TU<br>
                    ES
                </div>

                <div class="kath">
                    KATHARINEUM<br>
                    ZU LÜBECK
                </div>

                <div class="wheel">
                    ✥
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NAVIGATION
# ============================================================

def navigate(
    page
):

    st.session_state[
        "page"
    ] = page

    st.rerun()


if "page" not in st.session_state:

    st.session_state[
        "page"
    ] = "start"


# ============================================================
# STARTSEITE – ABB. 1
# ============================================================

def start_page():

    show_header()


    # --------------------------------------------------------
    # FOTO HOCHLADEN
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="big-button">

            <span>
                FOTO HOCHLADEN
            </span>

            <span class="big-button-icon">
                ⇧
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    if st.button(
        "FOTO AUFNEHMEN / HOCHLADEN",
        use_container_width=True
    ):

        navigate(
            "upload"
        )


    # --------------------------------------------------------
    # SUCHEN
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="big-button">

            <span>
                KLEIDUNGSSTÜCK SUCHEN
            </span>

            <span class="big-button-icon">
                ⌕
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    search_text = st.text_input(
        "Suche",
        placeholder="z.B.: Flaschen",
        label_visibility="collapsed"
    )


    if st.button(
        "SUCHEN",
        use_container_width=True
    ):

        st.session_state[
            "search_text"
        ] = search_text

        navigate(
            "search"
        )


    # --------------------------------------------------------
    # LETZTE FUNDSTÜCKE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            LETZTE FUNDSTÜCKE
        </div>
        """,
        unsafe_allow_html=True
    )


    items = get_latest(
        6
    )


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

            columns = st.columns(
                3,
                gap="small"
            )

            for column, item in zip(
                columns,
                items[start:start + 3]
            ):

                with column:

                    st.image(
                        item["bild"],
                        use_container_width=True
                    )

                    if st.button(
                        "→",
                        key=f"latest_{item['id']}",
                        use_container_width=True
                    ):

                        st.session_state[
                            "selected_id"
                        ] = item["id"]

                        navigate(
                            "detail"
                        )


    show_footer()


# ============================================================
# FOTO HOCHLADEN
# ============================================================

def upload_page():

    show_header()


    if st.button(
        "← ZURÜCK",
        use_container_width=True
    ):

        navigate(
            "start"
        )


    st.markdown(
        """
        <div class="section-title">
            FOTO HOCHLADEN
        </div>
        """,
        unsafe_allow_html=True
    )


    # Kamera

    camera_image = st.camera_input(
        "Foto aufnehmen"
    )


    # Datei

    uploaded_image = st.file_uploader(

        "oder Foto auswählen",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    image_source = (

        camera_image

        if camera_image is not None

        else uploaded_image
    )


    if image_source is None:

        st.info(
            "Nimm ein Foto auf oder "
            "wähle ein vorhandenes Bild."
        )

        show_footer()

        return


    image = Image.open(
        image_source
    )


    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------------
    # KI ERKENNUNG
    # --------------------------------------------------------

    try:

        with st.spinner(
            "KI erkennt das Fundstück ..."
        ):

            predicted, confidence, ranking = (
                classify_image(
                    image
                )
            )


    except Exception as error:

        st.error(
            "KI-Erkennung fehlgeschlagen:\n\n"
            f"{error}"
        )

        st.info(
            "Falls hier weiterhin ein "
            "Keras-Kompatibilitätsfehler erscheint, "
            "prüfe bitte die TensorFlow-Version."
        )

        show_footer()

        return


    st.markdown(
        f"""
        <div class="section-title">
            ERKANNT: {predicted.upper()}
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


    # --------------------------------------------------------
    # WEITERE KLASSEN
    # --------------------------------------------------------

    with st.expander(
        "KI-Ergebnisse anzeigen"
    ):

        for label, probability in ranking:

            st.write(
                f"{label}: {probability:.1%}"
            )


    # --------------------------------------------------------
    # DATEN EINGEBEN
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            DATEN EINGEBEN
        </div>
        """,
        unsafe_allow_html=True
    )


    labels = load_labels()


    with st.form(
        "fundstueck_form"
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


        name = st.text_input(
            "Bezeichnung",
            placeholder="z.B. blaue Flasche"
        )


        found_date = st.date_input(
            "Gefunden am",
            date.today()
        )


        found_location = st.text_input(
            "Fundort",
            placeholder="z.B. Obere Turnhalle"
        )


        notes = st.text_area(
            "Notizen",
            placeholder="Weitere Informationen ..."
        )


        current_location = st.text_input(
            "Aktueller Standort",
            placeholder="z.B. Fundkiste"
        )


        color = st.text_input(
            "Farbe",
            placeholder="z.B. blau"
        )


        submitted = st.form_submit_button(
            "FUNDSTÜCK SPEICHERN",
            use_container_width=True
        )


    if submitted:

        item_id = save_fundstueck(

            image,

            category,

            name,

            found_date.isoformat(),

            found_location,

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
            "Fundstück gespeichert."
        )


        if st.button(
            "ZUR STARTSEITE",
            use_container_width=True
        ):

            navigate(
                "start"
            )


    show_footer()


# ============================================================
# SUCHSEITE – ABB. 2
# ============================================================

def search_page():

    show_header()


    if st.button(
        "← ZURÜCK",
        use_container_width=True
    ):

        navigate(
            "start"
        )


    # --------------------------------------------------------
    # SUCHFELD WIE IN DER SKIZZE
    # --------------------------------------------------------

    search_text = st.session_state.get(
        "search_text",
        ""
    )


    search_text = st.text_input(

        "Suchbegriff",

        value=search_text,

        placeholder="z.B.: Flaschen"
    )


    all_items = get_latest(
        10000
    )


    categories = [
        "Alle"
    ] + sorted(
        {
            item["kategorie"]
            for item in all_items
        }
    )


    colors = [
        "Alle"
    ] + sorted(
        {
            item["farbe"]
            for item in all_items
            if item["farbe"]
        }
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        category = st.selectbox(
            "Kategorie",
            categories
        )


    with col2:

        color = st.selectbox(
            "Farbe",
            colors
        )


    results = search_fundstuecke(

        search_text,

        category,

        color
    )


    # --------------------------------------------------------
    # ERGEBNISÜBERSCHRIFT
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="results-title">

            ERGEBNISSE FÜR
            „{search_text.upper() if search_text else "ALLE"}“

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # 3-SPALTEN-GRID
    # --------------------------------------------------------

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

            columns = st.columns(
                3,
                gap="small"
            )


            for column, item in zip(
                columns,
                results[start:start + 3]
            ):

                with column:

                    st.image(
                        item["bild"],
                        use_container_width=True
                    )


                    # Der Button ist klein,
                    # damit das Layout der Skizze
                    # möglichst erhalten bleibt.

                    if st.button(
                        "→",
                        key=f"search_{item['id']}",
                        use_container_width=True
                    ):

                        st.session_state[
                            "selected_id"
                        ] = item["id"]

                        navigate(
                            "detail"
                        )


    show_footer()


# ============================================================
# DETAILSEITE – ABB. 3
# ============================================================

def detail_page():

    item = get_fundstueck(

        st.session_state.get(
            "selected_id"
        )
    )


    if item is None:

        navigate(
            "start"
        )


    show_header()


    # --------------------------------------------------------
    # ZURÜCK-PFEIL
    # --------------------------------------------------------

    if st.button(
        "←",
        use_container_width=False
    ):

        navigate(
            "search"
        )


    # --------------------------------------------------------
    # SUCHBEGRIFF / KATEGORIE
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="search-field">

            <span>
                {item["bezeichnung"]
                or
                item["kategorie"]}
            </span>

            <span class="search-icon">
                ⌕
            </span>

        </div>

        <div class="section-title">
            ERGEBNISSE FÜR
            „{item["kategorie"].upper()}“
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # GROSSES FOTO
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="detail-image">

            <img
                src="file://{Path(item["bild"]).resolve()}"
            >

        </div>
        """,
        unsafe_allow_html=True
    )


    # Streamlit Bild sicherheitshalber zusätzlich anzeigen,
    # falls der Browser file:// nicht darstellt.

    st.image(
        item["bild"],
        width=165
    )


    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="details">

            <div class="detail-row">

                <span class="detail-label">
                    GEFUNDEN AM:
                </span>

                {item["funddatum"] or "-"}

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

                {item["standort"] or "-"}

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ZUGEORDNET
    # --------------------------------------------------------

    if item["status"] == "Gefunden":

        if st.button(
            "FUNDSTÜCK ZUGEORDNET",
            use_container_width=True
        ):

            update_status(
                item["id"],
                "Zugeordnet"
            )

            # Nach Betätigung zurück zur Startseite,
            # genau wie in deiner Skizze.

            navigate(
                "start"
            )

    else:

        st.markdown(
            """
            <div class="assigned">
                FUNDSTÜCK ZUGEORDNET
            </div>
            """,
            unsafe_allow_html=True
        )


        if st.button(
            "ZUR STARTSEITE",
            use_container_width=True
        ):

            navigate(
                "start"
            )


    show_footer()


# ============================================================
# SEITEN-ROUTER
# ============================================================

page = st.session_state[
    "page"
]


if page == "start":

    start_page()


elif page == "upload":

    upload_page()


elif page == "search":

    search_page()


elif page == "detail":

    detail_page()


else:

    start_page()
