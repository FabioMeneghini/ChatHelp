import streamlit as st
from txtai import Embeddings
from chatbot import Chatbot
from document import Document

class GUIChat:
    def __init__(self, chatbot: Chatbot) -> None:
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
        embeddings = Embeddings({
            "path": self.chatbot.similarity_model.get_path(),
            "content": True,
            "functions": [
                {"name": "graph", "function": "graph.attribute"},
            ],
            "expressions": [
                {"name": "category", "expression": "graph(indexid, 'category')"},
                {"name": "topic", "expression": "graph(indexid, 'topic')"},
                {"name": "topicrank", "expression": "graph(indexid, 'topicrank')"}
            ],
            "graph": {
                "limit": 15,
                "minscore": 0.2
            }
        })
        msg = self.chatbot.regenerate_kg(embeddings)
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