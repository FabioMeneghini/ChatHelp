import time
from txtai import Embeddings
from db_access import DBAccess
from knowledge_graph import KnowledgeGraph
from question_answering_llm import QuestionAnsweringLLM
from rank_fusion import RankFusion
from similarity_llm import SimilarityLLM
from zero_shot_llm import ZeroShotLLM

class Chatbot:
    violations_dict = {
        "S1": "violenza",
        "S2": "reati non violenti",
        "S3": "violenza sessuale",
        "S4": "abuso di minori",
        "S5": "diffamazione",
        "S6": "consulenza specializzata",
        "S7": "informazioni personali",
        "S8": "violazione di proprietà intellettuale",
        "S9": "costruzione di armi",
        "S10": "odio",
        "S11": "autolesionismo",
        "S12": "contenuti per adulti",
        "S13": "disinformazione",
        "S14": "codice malevolo"
    }

    def _msg_stream(self, text):
        for word in text.split(" "):
            yield word+' '
            time.sleep(0.025)
    
    def __init__(self, connection: DBAccess, qa_model: QuestionAnsweringLLM, zs_model: ZeroShotLLM,
                 similarity_model: SimilarityLLM, kg: KnowledgeGraph, rank_fusion: RankFusion):
        self.connection = connection
        self.qa_model = qa_model
        self.zs_model = zs_model
        self.similarity_model = similarity_model
        self.kg = kg
        #self.kg.create_index("./index")
        self.kg.load_index("./index")
        self.rank_fusion = rank_fusion
    
    def set_rank_fusion(self, strategy: RankFusion):
        self.rank_fusion = strategy

    def _build_query(ids): # crea una query per cercare i nodi collegati ai nodi con gli id passati
        query = f'MATCH P=({{id: "{ids[0]}"}})'
        for i in range(1, len(ids)):
            query += f'-[*1..10]->({{id: "{ids[i]}"}})' # massimo 10 nodi intermedi tra quelli passati
        query += ' RETURN P LIMIT 3' # ritorna i 3 migliori percorsi trovati
        return query

    def get_qa_model_name(self):
        return self.qa_model.get_name()

    def set_qa_model(self, qa_model: QuestionAnsweringLLM):
        self.qa_model = qa_model
    
    def regenerate_kg(self, embeddings: Embeddings, path="./index"):
        yield "Sto rigenerando il Knowledge Graph...\n\n"
        yield self.kg.regenerate(embeddings, path)
    
    def generate_response(self, prompt, use_kg) -> str:
        safety = self._check_safety(prompt)
        if safety != "safe":
            return self._msg_stream(safety)
        bm25_results = self.connection.get_bm25_rank(prompt) # ricerca con BM25
        embeddings_results = self.connection.get_embeddings_rank(prompt, self.similarity_model) # ricerca semantica
        
        if bm25_results is None or not isinstance(bm25_results, list) or embeddings_results is None or not isinstance(embeddings_results, list):
            return self._msg_stream("Nessuna risposta generata.")
        elif len(bm25_results) == 0 or len(embeddings_results) == 0: # se almeno una delle due liste non ha risultati
            return self._msg_stream(f"Nessun risultato trovato per la richiesta fornita ('{prompt}').")
        else:
            embeddings_results = [(t[0], 1 - t[1], t[2], t[3]) for t in embeddings_results] # inverte il punteggio della ricerca semantica per avere un ranking crescente
            results = self.rank_fusion.fuse(bm25_results, embeddings_results) # fonde i risultati delle due ricerche con il metodo scelto

            best_results = [result for result in results if result[1] > 0.6] # prendi solo i risultati con un punteggio alto
            best_results = best_results[:6] # prendi solo i primi 6 risultati
            
            if len(best_results) < 3: # se non ci sono almeno 3 risultati con un punteggio alto, prendi i primi 3
                best_results = results[:3]

            find_sections = False
            if use_kg: # se è stata selezionata la ricerca con Knowledge Graph
                best_results = best_results[:4] # prendi solo i primi 4 risultati per non rallentare troppo
                best_results = self.kg.search(Chatbot._build_query([result[0] for result in best_results]))
                find_sections = True
            
            best_results = sorted(best_results, key=lambda x: int(x[0])) # ordina i risultati in base al loro codice

            context = "\n".join([result[2] for result in best_results]) # crea il contesto recuperando i testi dei risultati migliori
            question = prompt
            response = self.qa_model.answer(question, context) # risposta del LLM basata sul contesto
            response += self._display_source(response, best_results, find_sections) # visualizza le fonti dei risultati migliori
            return self._msg_stream(response)
    
    def _check_safety(self, prompt):
        check = self.qa_model.check(prompt)
        if check.startswith("unsafe"): # controlla se la domanda è sicura
            violations = check.split('\n')[1]
            response = "Non posso rispondere a questa richiesta. Motivo:\nla richiesta include"
            for word in violations.split(' '): # stampa le parole pericolose
                response += " "+Chatbot.violations_dict[word]+","
            response = response[:-1] + "."
            return response
        else:
            return "safe"
    
    def _display_source(self, response, best_results, find_sections):
        source_msg = ""
        if not find_sections:
            sources = [result[3] for result in best_results] # prende le fonti dei risultati migliori
        else:
            sources = self.connection.get_sections_by_code([result[0] for result in best_results]) # prende le sezioni dei risultati migliori
            sources = [s[0] for s in sources]
        sources = list(set(sources)) # rimuove duplicati
        label = self.zs_model.classify(response) # classifica la risposta del LLM
        if label == "impossibile":
            return ""
        source_msg += "\n\nLe informazioni fornite sono state ritrovate nelle sezioni: "
        source_msg += f"***{sources[0]}***"
        for s in sources[1:]:
            source_msg += f', ***{s}***'
        source_msg += "."
        return source_msg