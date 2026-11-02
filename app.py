# ============================================================
# UPLOAD-SEITE
# ============================================================

def page_upload():

    show_header(
        show_back=True
    )

    st.subheader(
        "FUNDSTÜCK AUFNEHMEN"
    )

    # ========================================================
    # VORHANDENES FOTO AUSWÄHLEN
    # KEINE KAMERA-FUNKTION
    # ========================================================

    selected_image = st.file_uploader(
        "Vorhandenes Foto auswählen",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="item_image_upload"
    )

    if selected_image is not None:

        image = Image.open(
            selected_image
        ).convert("RGB")

        st.image(
            image,
            width="stretch"
        )

        # ====================================================
        # KI-ERKENNUNG
        # ====================================================

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

            with st.expander(
                "Weitere Ergebnisse"
            ):

                for result in results:

                    result_name = clean_label(
                        result["label"]
                    )

                    result_confidence = (
                        result["confidence"]
                        * 100
                    )

                    st.write(
                        f"{result_name}: "
                        f"{result_confidence:.1f} %"
                    )

        # ====================================================
        # INFORMATIONEN
        # ====================================================

        st.subheader(
            "INFORMATIONEN"
        )

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

        st.write("")

        # ====================================================
        # FUNDSTÜCK SPEICHERN
        # ====================================================

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
                    image_path=image_path,
                    name=name,
                    kategorie=category,
                    farbe=color,
                    gefunden_am=found_date.isoformat(),
                    fundort=location,
                    notizen=notes,
                    aktueller_standort=current_location
                )

                st.session_state.selected_item = (
                    item_id
                )

                go_to("detail")

    show_footer()
