import streamlit as st

class GUISidebar:
    def __init__(self, connection) -> None:
        self.radio_llm = None
        self.radio_rf = None
        self.toggle_kg = None
        self.btn_idf = False
        self.btn_emb = False
        self.btn_kg = False
        self.btn_new_doc = False
        self.file_up = None
        self.connection = connection

        with st.sidebar: # menu nella barra laterale
            self.show_base()
            with st.expander("Funzionalità avanzate"):
                if st.session_state.is_admin: # se l'utente ha fatto l'accesso come amministratore
                    self.show_advanced()
                else:
                    self.show_form()

    def show_form(self):
        password = st.text_input("Password", type="password")
        btn_admin = st.button("Accedi come amministratore")
        if btn_admin and password == "admin":
            st.session_state.update(is_admin=True)
            st.rerun()
        elif btn_admin and password != "admin":
            st.error("Password errata")

    def show_advanced(self):
        self.file_up = st.file_uploader("Carica un nuovo documento", type=["pdf", "sam"])
        self.btn_new_doc = st.button("Carica documento", type="primary", disabled=self.file_up is None)
        st.markdown("---")
        self.btn_idf = st.button("Ricalcola IDF", type="primary")
        self.btn_emb = st.button("Ricalcola vettori di embedding", type="primary")
        self.btn_kg = st.button("Rigenera Knowledge Graph", type="primary")
        st.button("Esci dalla modalità admin", on_click=lambda: st.session_state.pop("is_admin", None))

    def show_base(self):
        st.subheader("Documento caricato")
        st.markdown(f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; font-style: italic;">
                {self.connection.get_document_name()}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.subheader("Modelli")
        self.radio_llm = st.radio("Seleziona il Large Language Model per la generazione del testo", ("Mixtral 8x7B", "Meta Llama 3 8B", "Gemma 2 9B"))
        st.subheader("Ricerca")
        self.radio_rf = st.radio("Seleziona l'algoritmo di Rank Fusion", ("Distribution-Based Score Fusion", "Reciprocal Rank Fusion"))
        self.toggle_kg = st.checkbox("Ricerca con Knowledge Graph", help="Attenzione: potrebbe rallentare considerevolmente il tempo di risposta.")
        st.subheader("Altro")
        pulizia_disabled = "messages" not in st.session_state or len(st.session_state.messages) == 0
        st.button("Pulisci cronologia chat", on_click=lambda: st.session_state.pop("messages", None), disabled=pulizia_disabled)
    
    def get_radio_llm(self):
        return self.radio_llm

    def get_radio_rf(self):
        return self.radio_rf
    
    def get_toggle_kg(self):
        return self.toggle_kg
    
    def get_btn_idf(self):
        return self.btn_idf
    
    def get_btn_emb(self):
        return self.btn_emb
    
    def get_btn_kg(self):
        return self.btn_kg

    def get_btn_new_doc(self):
        return self.btn_new_doc

    def get_file_up(self):
        return self.file_up