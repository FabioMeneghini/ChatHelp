import os
import sys
sys.path.append('src')
from db_access import DBAccess
from similarity_llm import SimilarityLLM
from pdf_document import PDFDocument

def test_connect_1(): #credenziali database non corrette
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 99999)
        assert db.outcome.startswith("Errore di connessione al database:")
    finally:
        db.close_connection()

def test_connect_2(): #credenziali database corrette
    db = DBAccess("localhost", "esempi", "postgres", "", 5432)
    assert db.outcome == "Connessione al database riuscita."
    
def test_connect_3(): #connessione già aperta
    try:
        db = DBAccess("localhost", "esempi", "postgres", "", 5432)
        assert db.connect() == "Connessione già aperta."
    finally:
        db.close_connection()

def test_close_connection(): #connessione chiusa
    db = DBAccess("localhost", "esempi", "postgres", "", 5432)
    db.close_connection()
    assert db.outcome == "Connessione al database chiusa."

def test_get_bm25_rank_1(): #test funzione get_bm25_rank no connessione
    try:
        db_1 = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        result = db_1.get_bm25_rank("test")
        assert result == None
        db = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette ma connesione chiusa
        db.close_connection()
        result = db.get_bm25_rank("test")
        assert result == None
    finally:
        db_1.close_connection()
        db.close_connection()

def test_get_bm25_rank_2(): #test funzione get_bm25_rank
    try:
        db = DBAccess("localhost", "documentazione", "postgres", "", 5432) #credenziali database corrette
        result = db.get_bm25_rank("cane")
        assert isinstance(result, list)
    finally:
        db.close_connection()

"""
def test_get_bm25_rank_3(): #test funzione get_bm25_rank
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 9999) #credenziali database corrette ma connesione chiusa
        result = db.get_bm25_rank("cane")
        assert result.startswith("Errore nel calcolo del rank BM25:")
    finally:
        db.close_connection()
"""

def test_get_embeddings_rank_1(): #test funzione get_embeddings_rank
    model = SimilarityLLM("nickprock/sentence-bert-base-italian-xxl-uncased")
    try:
        db = DBAccess("localhost", "documentazione", "postgres", "", 5432) #credenziali database corrette
        result = db.get_embeddings_rank("cane", model)
        assert isinstance(result, list)
    finally:
        db.close_connection()

def test_get_embeddings_rank_2(): #test funzione get_embeddings_rank no connessione
    model = SimilarityLLM("nickprock/sentence-bert-base-italian-xxl-uncased")
    try:
        db_1 = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        result = db_1.get_embeddings_rank("cane", model)
        assert result == None
        db = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette ma connesione chiusa
        db.close_connection()
        result = db.get_embeddings_rank("cane", model)
        assert result == None
    finally:
        db_1.close_connection()
        db.close_connection()


def test_insert_new_document_1(): #test funzione insert_new_document no connessione
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        out = list(db.insert_new_document(PDFDocument("docs/Infinity_CRM.pdf")))
        assert out[0] == "Non connesso al database."
    finally:
        db.close_connection()

def test_insert_new_document_2(): #test funzione insert_new_document
    try:
        db = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette, ma database sbagliato
        out = list(db.insert_new_document(PDFDocument("docs/Infinity_CRM.pdf")))
        assert out[0].startswith("Errore durante l'inserimento del documento:")
    finally:
        db.close_connection()

def test_insert_new_document_3(): #test funzione insert_new_document
    try:
        file_name = "Infinity_CRM.pdf"
        os.makedirs('./docs', exist_ok=True)
        save_path = os.path.join("./docs", file_name)
        db = DBAccess("localhost", "documentazione", "postgres", "", 5432) #credenziali database corrette, ma database sbagliato
        document = PDFDocument(save_path)
        out = list(db.insert_new_document(document))
        print(out)
        assert out[0] == "Sto inserendo il nuovo documento...\n\n"
        assert out[1] == "Documento inserito con successo."
    finally:
        db.close_connection()

def test_calculate_embeddings_1(): #test funzione calculate_embeddings no connessione
    model = SimilarityLLM("nickprock/sentence-bert-base-italian-xxl-uncased")
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        out = list(db.calculate_embeddings(model))
        assert out[0] == "Non connesso al database."
        db_1 = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette ma connesione chiusa   
        db_1.close_connection()
        out = list(db_1.calculate_embeddings(model))
        assert out[0] == "Non connesso al database."
    finally:
        db.close_connection()
        db_1.close_connection()

def test_calculate_embeddings_2(): #test funzione calculate_embeddings
    model = SimilarityLLM("nickprock/sentence-bert-base-italian-xxl-uncased")
    try:
        db = DBAccess("localhost", "documentazione", "postgres", "", 5432) #credenziali database corrette
        out = list(db.calculate_embeddings(model))
        assert out[0] == "Sto calcolando gli embeddings...\n\n"
        assert out[1] == "Embeddings ricalcolati e salvati con successo."
    finally:
        db.close_connection()

"""
def test_calculate_embeddings_3(): #test funzione calculate_embeddings
    model = SimilarityLLM("xxxxx")
    try:
        db = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette
        out = list(db.calculate_embeddings(model))
        assert out[0].startswith("Sto calcolando gli embeddings...\n\n")
        assert out[1].startswith("Errore durante il calcolo degli embeddings:")
    finally:
        db.close_connection()
"""

def test_calculate_idf_1(): #test funzione calculate_idf no connessione
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        out = list(db.calculate_idf())
        assert out[0] == "Non connesso al database."
    finally:
        db.close_connection()

def test_calculate_idf_2(): #test funzione calculate_idf
    try:
        db = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette
        out = list(db.calculate_idf())
        assert out[0] == "Sto calcolando gli IDF...\n\n"
        assert out[1] == "IDF ricalcolati e salvati con successo."
    finally:
        db.close_connection()

def test_get_all_rows_1(): #test funzione get_all_rows no connessione
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        out = db.get_all_rows()
        assert out == "Non connesso al database."
    finally:
        db.close_connection()

def test_get_all_rows_2(): #test funzione get_all_rows
    try:
        db = DBAccess("localhost", "documentazione", "postgres", "", 5432) #credenziali database corrette
        out = db.get_all_rows()
        assert isinstance(out, list)
    finally:
        db.close_connection()

def test_get_document_name_1(): #test funzione get_document_name no connessione
    try:
        db = DBAccess("xxx", "xxx", "xxx", "xxx", 99999) #credenziali database non corrette
        out = db.get_document_name()
        assert out == "Non connesso al database."
    finally:
        db.close_connection()

def test_get_document_name_2(): #test funzione get_document_name
    try:
        db = DBAccess("localhost", "esempi", "postgres", "", 5432) #credenziali database corrette ma database sbagliato
        out = db.get_document_name()
        assert out.startswith("Errore durante il recupero del nome del documento:")
    finally:
        db.close_connection()

##################### DA ESEGUIRE DOPO test_insert_new_document_3 #####################
def test_get_document_name_3(): #test funzione get_document_name
    try:
        db = DBAccess("localhost", "documentazione", "postgres", "", 5432) #credenziali database corrette
        out = db.get_document_name()
        assert out == "Infinity_CRM.pdf"
    finally:
        db.close_connection()
