from collections import defaultdict
from rank_fusion import RankFusion

class RRFRankFusion(RankFusion):
    def fuse(self, *ranks_lists):
        return self._rrf(*ranks_lists)
    
    def _rrf(self, *ranks_lists, k=60, normalize=True): # metodo di rank fusion Reciprocal Rank Fusion: si basa esclusivamente sull'ordine dei documenti nei ranking passati come parametri, non guarda i punteggi
        rrf_map = defaultdict(float)
        item_details = {}
        for rank_list in ranks_lists:
            for rank, item in enumerate(rank_list, 1):
                codice = item[0]
                testo = item[2]
                sezione = item[3]
                rrf_map[codice] += 1 / (rank + k)
                item_details[codice] = (codice, testo, sezione)  # Memorizza codice e testo per l'uso finale
        # Crea una lista di tuple con codice, testo e punteggio RRF
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        result = [(item_details[codice][0], score, item_details[codice][1], item_details[codice][2]) for codice, score in sorted_items]
        if normalize: #normalizza i punteggi tra 0 e 1 con min-max scaling
            max_score = max(result, key=lambda x: x[1])[1]
            min_score = min(result, key=lambda x: x[1])[1]
            for i in range(len(result)):
                if max_score == min_score:
                    result[i] = (result[i][0], 1.0, result[i][2], result[i][3])
                else:
                    result[i] = (result[i][0], (result[i][1] - min_score) / (max_score - min_score), result[i][2], result[i][3])
        return result #(codice, punteggio normalizzato, testo, sezione)