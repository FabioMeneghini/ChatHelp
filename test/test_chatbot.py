import sys
sys.path.append('src')
from txtai import Embeddings
from knowledge_graph import KnowledgeGraph
from chatbot import Chatbot
from db_access import DBAccess
from dbsf_rank_fusion import DBSFRankFusion
from question_answering_llm import QuestionAnsweringLLM
from zero_shot_llm import ZeroShotLLM
from similarity_llm import SimilarityLLM

zs_model = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
similarity_model = SimilarityLLM("nickprock/sentence-bert-base-italian-xxl-uncased")
emb = Embeddings({
                "path": similarity_model.get_path(),
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

def test_get_qa_model_name():
    connection = None
    qa_model = QuestionAnsweringLLM("mixtral-8x7b-32768")
    rank_fusion = DBSFRankFusion()
    knowledge_graph = KnowledgeGraph(connection, emb)
    chatbot = Chatbot(connection, qa_model, zs_model, similarity_model, knowledge_graph, rank_fusion)
    assert chatbot.get_qa_model_name() == "mixtral-8x7b-32768"

def test_set_qa_model():
    connection = None
    qa_model = QuestionAnsweringLLM("mixtral-8x7b-32768")
    rank_fusion = DBSFRankFusion()
    knowledge_graph = KnowledgeGraph(connection, emb)
    chatbot = Chatbot(connection, qa_model, zs_model, similarity_model, knowledge_graph, rank_fusion)
    qa_model_1 = QuestionAnsweringLLM("llama3-8b-8192")
    chatbot.set_qa_model(qa_model_1)
    assert chatbot.get_qa_model_name() == "llama3-8b-8192"

def test_regenerate_kg_1(): #grafo non rigenerato
    connection = None
    qa_model = QuestionAnsweringLLM("mixtral-8x7b-32768")
    rank_fusion = DBSFRankFusion()
    knowledge_graph = KnowledgeGraph(connection, emb)
    chatbot = Chatbot(connection, qa_model, zs_model, similarity_model, knowledge_graph, rank_fusion)
    out = chatbot.regenerate_kg(emb, "../index")
    out = list(out)
    assert len(out) == 2
    assert out[0] == "Sto rigenerando il Knowledge Graph...\n\n"
    assert out[1].startswith("Errore durante la rigenerazione del grafo:")

def test_regenerate_kg_2(): #grafo rigenerato
    try:
        db_access = DBAccess("localhost", "documentazione", "postgres", "", "5432")
        qa_model = QuestionAnsweringLLM("mixtral-8x7b-32768")
        rank_fusion = DBSFRankFusion()
        knowledge_graph = KnowledgeGraph(db_access, emb)
        chatbot = Chatbot(db_access, qa_model, zs_model, similarity_model, knowledge_graph, rank_fusion)
        out = chatbot.regenerate_kg(emb, "../index")
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
        qa_model = QuestionAnsweringLLM("llama3-8b-8192")
        rank_fusion = DBSFRankFusion()
        knowledge_graph = KnowledgeGraph(db_access, emb)
        chatbot = Chatbot(db_access, qa_model, zs_model, similarity_model, knowledge_graph, rank_fusion)
        out = list(chatbot.generate_response("prova", False))
        #print(out)
        assert out[0] != "Nessuna risposta generata." and not out[0].startswith("Nessun risultato trovato per la richiesta fornita")
    finally:
        db_access.close_connection()