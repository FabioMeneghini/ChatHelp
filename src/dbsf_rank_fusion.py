from collections import defaultdict
import itertools
from math import inf
from rank_fusion import RankFusion

class DBSFRankFusion(RankFusion):
    def fuse(self, *ranks_lists): # metodo di rank fusion DBSF: fonde i ranking passati basandosi sui punteggi dei documenti normalizzati con min-max scaling basato su media e deviazione standard
        dbsf_rank_lists = []
        for rank_list in ranks_lists:
            if len(rank_list) == 1:
                dbsf_rank_lists.append(rank_list)
                continue
            mean_score = sum(doc[1] for doc in rank_list) / len(rank_list)
            std_dev = (sum((doc[1] - mean_score) ** 2 for doc in rank_list) / len(rank_list)) ** 0.5
            min_score = mean_score - 3 * std_dev
            max_score = mean_score + 3 * std_dev

            dbsf_rank_list = []
            for doc in rank_list:
                if max_score == min_score:
                    dbsf_rank_list.append((doc[0], 1.0, doc[2], doc[3]))
                else:
                    dbsf_rank_list.append((doc[0], (doc[1] - min_score) / (max_score - min_score), doc[2], doc[3]))
            dbsf_rank_lists.append(dbsf_rank_list)
        
        output = []
        docs_per_id = defaultdict(list)
        for doc in itertools.chain.from_iterable(dbsf_rank_lists):
            docs_per_id[doc[0]].append(doc)
        for docs in docs_per_id.values():
            doc_with_best_score = max(docs, key=lambda doc: doc[1] if doc[1] else -inf)
            output.append(doc_with_best_score)
        output.sort(key=lambda doc: doc[1], reverse=True)
        return output