import streamlit as st
from document import Document

class GUIChat:
    def __init__(self, chatbot) -> None:
        self.chatbot = chatbot
    
    def show_message(self, msg_stream):
        with st.chat_message("assistant", avatar="🤖"):
            response = st.write_stream(msg_stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    def show_response_message(self, prompt, use_kg):
        msg = self.chatbot.generate_response(prompt, use_kg)
        self.show_message(msg)
    
    def show_btn_idf_message(self):
        msg = self.chatbot.connection.calculate_idf()
        self.show_message(msg)

    def show_btn_emb_message(self):
        msg = self.chatbot.connection.calculate_embeddings(self.chatbot.similarity_model)
        self.show_message(msg)
    
    def show_btn_kg_message(self):
        msg = self.chatbot.regenerate_kg()
        self.show_message(msg)
    
    def show_btn_new_doc_message(self, document: Document):
        msg = st.session_state.connection.insert_new_document(document)
        self.show_message(msg)
        self.show_btn_idf_message()
        self.show_btn_emb_message()
        self.show_btn_kg_message()
    
    def show_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
    
    def get_chatbot(self):
        return self.chatbot