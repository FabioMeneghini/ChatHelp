#from datasets import load_dataset
from txtai import Embeddings
import matplotlib.pyplot as plt
import networkx as nx
from db_access import DBAccess

class KnowledgeGraph:
    def __init__(self, connection: DBAccess, embeddings: Embeddings):
        self.setup(connection, embeddings)
    
    def setup(self, connection: DBAccess, embeddings: Embeddings):
        self.connection = connection
        self.graph = None
        self.embeddings = embeddings
    
    def regenerate(self, embeddings: Embeddings, save_path): # rigenera il grafo
        try:
            self.close_index()
            self.setup(self.connection, embeddings)
            self.create_index(save_path)
            return "Knowledge Graph rigenerato con successo."
        except Exception as e:
            return f"Errore durante la rigenerazione del grafo: {e}"

    def _stream(self): # restituisce un generatore di tuple (id, testo) per ogni riga del database
        results = self.connection.get_all_rows()
        for row in results:
            yield (f"{row[0]}", f"{row[1]}")
    
    def load_index(self, path): # carica l'indice dal percorso passato come parametro
        self.embeddings.load(path)
    
    def close_index(self): # chiude l'indice
        self.embeddings.close()
    
    def create_index(self, path): # crea l'indice e lo salva nel percorso passato come parametro (potrebbe essere un'operazione lunga)
        self.embeddings.index(self._stream())
        self.embeddings.save(path)

    def search(self, query): # esegue una ricerca nel grafo e restituisce i risultati
        self.graph = self.embeddings.graph.search(query, graph=True)
        # for node in list(self.graph.scan()):
        #     print(node)
        results = [(self.graph.attribute(node, 'id'), 0, self.graph.attribute(node, 'text'), "Sezione") for node in list(self.graph.scan())]
        return results

    def plot(self): # visualizza il grafo con matplotlib
        labels = {x: f"{self.graph.attribute(x, 'id')} ({x})" for x in self.graph.scan()}
        options = {
            "node_size": 1500,
            "node_color": "#0277bd",
            "edge_color": "#454545",
            #"font_color": "#fff",
            "font_color": "#000000",
            "font_size": 9,
            "alpha": 1.0
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        pos = nx.spring_layout(self.graph.backend, seed=0, k=0.9, iterations=50)
        nx.draw_networkx(self.graph.backend, pos=pos, labels=labels, **options)
        ax.set_facecolor("#303030")
        ax.axis("off")
        fig.set_facecolor("#FFFFFF")
        plt.show()