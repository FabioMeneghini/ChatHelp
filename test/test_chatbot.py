import sys
sys.path.append('src')
from chatbot import Chatbot
from db_access import DBAccess
from dbsf_rank_fusion import DBSFRankFusion

def test_get_qa_model_name():
    chatbot = Chatbot(None,
                      "mixtral-8x7b-32768",
                      "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                      "nickprock/sentence-bert-base-italian-xxl-uncased")
    assert chatbot.get_qa_model_name() == "mixtral-8x7b-32768"

def test_set_qa_model():
    chatbot = Chatbot(None,
                      "mixtral-8x7b-32768",
                      "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                      "nickprock/sentence-bert-base-italian-xxl-uncased")
    chatbot.set_qa_model("llama3-8b-8192")
    assert chatbot.get_qa_model_name() == "llama3-8b-8192"

def test_regenerate_kg_1(): #grafo non rigenerato
    chatbot = Chatbot(None,
                      "mixtral-8x7b-32768",
                      "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                      "nickprock/sentence-bert-base-italian-xxl-uncased")
    out = chatbot.regenerate_kg()
    out = list(out)
    assert len(out) == 2
    assert out[0] == "Sto rigenerando il Knowledge Graph...\n\n"
    assert out[1].startswith("Errore durante la rigenerazione del grafo:")

def test_regenerate_kg_2(): #grafo rigenerato
    try:
        db_access = DBAccess("localhost", "documentazione", "postgres", "", "5432")
        chatbot = Chatbot(db_access,
                        "mixtral-8x7b-32768",
                        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        "nickprock/sentence-bert-base-italian-xxl-uncased")
        out = chatbot.regenerate_kg("../index")
        #db_access.close_connection()
        out = list(out)
        #print(out)
        assert len(out) == 2 
        assert out[0] == "Sto rigenerando il Knowledge Graph...\n\n" 
        assert out[1] == "Knowledge Graph rigenerato con successo."
    finally:
        db_access.close_connection()
    
def test_generate_response_1(): #risposta generata
    try:
        db_access = DBAccess("localhost", "documentazione", "postgres", "", "5432") #database esistente
        chatbot = Chatbot(db_access,
                        "llama3-8b-8192",
                        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                        "nickprock/sentence-bert-base-italian-xxl-uncased")
        chatbot.set_rank_fusion(DBSFRankFusion())
        out = list(chatbot.generate_response("prova", False))
        #print(out)
        assert out[0] != "Nessuna risposta generata." and not out[0].startswith("Nessun risultato trovato per la richiesta fornita")
    finally:
        db_access.close_connection()