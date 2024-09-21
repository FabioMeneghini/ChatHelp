import sys
sys.path.append('src')
from rrf_rank_fusion import RRFRankFusion
from dbsf_rank_fusion import DBSFRankFusion

def test_fuse_rrf_1(): #test situazione banale rrf
    rf = RRFRankFusion()
    rrf = rf.fuse([(1, 0.5, "a", "a"), (2, 0.4, "b", "b"), (3, 0.3, "c", "c")],
                  [(1, 0.6, "a", "a"), (2, 0.5, "b", "b"), (3, 0.4, "c", "c")])
    rrf_test = [(t[0], t[2]) for t in rrf]
    assert rrf_test == [(1, "a"), (2, "b"), (3, "c")]

def test_fuse_rrf_2(): #test situazione tipica rrf
    rf = RRFRankFusion()
    rrf = rf.fuse([(2, 0.5, "b", "b"), (3, 0.4, "c", "c"), (5, 0.3, "e", "e"), (1, 0.2, "a", "a"), (4, 0.1, "d", "d")],
                  [(3, 1.0, "c", "c"), (5, 0.9, "e", "e"), (2, 0.8, "b", "b"), (1, 0.7, "a", "a"), (4, 0.6, "d", "d")],
                  [(4, 2.0, "d", "d"), (2, 1.6, "b", "b"), (5, 1.2, "e", "e"), (3, 0.8, "c", "c"), (1, 0.4, "a", "a")])
    rrf_test = [(t[0], t[2]) for t in rrf]
    assert rrf_test == [(2, "b"), (3, "c"), (5, "e"), (4, "d"), (1, "a")]

def test_fuse_rrf_3(): #test un solo elemento rrf
    rf = RRFRankFusion()
    rrf = rf.fuse([(1, 0.2, "a", "a")])
    rrf_test = [(t[0], t[2]) for t in rrf]
    assert rrf_test == [(1, "a")]

def test_fuse_rrf_4(): #test lista con tutti punteggi uguali rrf
    rf = RRFRankFusion()
    rrf = rf.fuse([(1, 0.2, "a", "a"), (2, 0.2, "b", "b")])
    assert rrf == [(1, 1.0, "a", "a"), (2, 0.0, "b", "b")]

def test_fuse_dbsf_1(): #test situazione banale dbsf
    rf = DBSFRankFusion()
    dbsf = rf.fuse([(1, 0.5, "a", "a"), (2, 0.4, "b", "b"), (3, 0.3, "c", "c")],
                   [(1, 0.6, "a", "a"), (2, 0.5, "b", "b"), (3, 0.4, "c", "c")])
    dbsf_test = [(t[0], t[2]) for t in dbsf]
    assert dbsf_test == [(1, "a"), (2, "b"), (3, "c")]

def test_fuse_dbsf_2(): #test situazione tipica dbsf
    rf = DBSFRankFusion()
    dbsf = rf.fuse([(2, 0.2, "b", "b"), (3, 0.1, "c", "c"), (5, 0.09, "e", "e"), (1, 0.08, "a", "a"), (4, 0.07, "d", "d")],
                    [(3, 2.0, "c", "c"), (5, 0.95, "e", "e"), (2, 0.9, "b", "b"), (1, 0.85, "a", "a"), (4, 0.1, "d", "d")])
    dbsf_test = [(t[0], t[2]) for t in dbsf]
    assert dbsf_test == [(2, "b"), (3, "c"), (5, "e"), (1, "a"), (4, "d")]

def test_fuse_dbsf_3(): #test un solo elemento dbsf
    rf = DBSFRankFusion()
    dbsf = rf.fuse([(1, 0.2, "a", "a")])
    dbsf_test = [(t[0], t[2]) for t in dbsf]
    assert dbsf_test == [(1, "a")]

def test_fuse_dbsf_4(): #test lista con tutti punteggi uguali dbsf
    rf = DBSFRankFusion()
    dbsf = rf.fuse([(1, 0.2, "a", "a"), (2, 0.2, "b", "b")])
    assert dbsf == [(1, 1.0, "a", "a"), (2, 1.0, "b", "b")]