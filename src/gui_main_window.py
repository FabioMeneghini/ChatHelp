import os
import streamlit as st
from db_access import DBAccess
from chatbot import Chatbot
from dbsf_rank_fusion import DBSFRankFusion
from gui_sidebar import GUISidebar
from gui_chat import GUIChat
from pdf_document import PDFDocument
from rrf_rank_fusion import RRFRankFusion
from sam_document import SAMDocument

class MainWindow:
    llm_dict = {
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Meta Llama 3 8B": "llama3-8b-8192",
        "Gemma 2 9B": "gemma2-9b-it"
    }
    rank_fusion_dict = {
        "Distribution-Based Score Fusion": "dbsf",
        "Reciprocal Rank Fusion": "rrf"
    }

    def __init__(self) -> None:
        st.set_page_config(page_title="Chatbot", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")
        st.title("Chatbot")
        if "connection" not in st.session_state:
            st.session_state.connection = DBAccess("localhost", "documentazione", "postgres", "", "5432")
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = Chatbot(st.session_state.connection,
                                            "mixtral-8x7b-32768",
                                            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                                            "nickprock/sentence-bert-base-italian-xxl-uncased")
        if "is_admin" not in st.session_state:
            st.session_state.is_admin = False
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "document" not in st.session_state:
            st.session_state.document = None
        #self.document = None

    def update(self): # costruzione dell'interfaccia grafica
        sb = GUISidebar(st.session_state.connection)

        # Aggiorna il modello di question answering se è stato selezionato un modello diverso
        if sb.get_radio_llm() != st.session_state.chatbot.get_qa_model_name():
            st.session_state.chatbot.set_qa_model(MainWindow.llm_dict[sb.get_radio_llm()])
        
        chat = GUIChat(st.session_state.chatbot)
        chat.show_history()
        
        if st.session_state.is_admin: # se l'utente è amministratore e ha cliccato uno dei pulsanti
            if sb.get_btn_idf():
                chat.show_btn_idf_message()
            if sb.get_btn_emb():
                chat.show_btn_emb_message()
            if sb.get_btn_kg():
                chat.show_btn_kg_message()
            if sb.get_btn_new_doc():
                os.makedirs('./docs', exist_ok=True)
                save_path = os.path.join("./docs", sb.get_file_up().name)
                with open(save_path, 'wb') as f:
                    f.write(sb.get_file_up().getbuffer())
                if sb.get_file_up().name.endswith(".pdf"):
                    st.session_state.document = PDFDocument(save_path)
                elif sb.get_file_up().name.endswith(".sam"):
                    st.session_state.document = SAMDocument(save_path)
                #elif sb.get_file_up().name.endswith(".md"):
                #    document = MDDocument(save_path)
                else:
                    st.error("Formato non supportato")
                    st.stop()
                chat.show_btn_new_doc_message(st.session_state.document)
                st.rerun()
        
        if prompt := st.chat_input("Inserisci una richiesta..."): # se è stata inserita una richiesta dall'utente
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            if MainWindow.rank_fusion_dict[sb.get_radio_rf()] == "dbsf":
                rank_fusion = DBSFRankFusion()
            elif MainWindow.rank_fusion_dict[sb.get_radio_rf()] == "rrf":
                rank_fusion = RRFRankFusion()
            else:
                st.error("Algoritmo di Rank Fusion non supportato")
                st.stop()
            chat.get_chatbot().set_rank_fusion(rank_fusion)
            chat.show_response_message(prompt, sb.get_toggle_kg())

if __name__ == "__main__":
    w = MainWindow()
    w.update()