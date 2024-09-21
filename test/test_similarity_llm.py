import sys
import huggingface_hub
import pytest
sys.path.append('src')
from similarity_llm import SimilarityLLM

def test_get_vector(): #testa che la funzione get_vector restituisca un vettore di 768 elementi
    model = SimilarityLLM("nickprock/sentence-bert-base-italian-xxl-uncased")
    vettore = model.get_vector("test")
    assert len(vettore) == 768

def test_exception():
    with pytest.raises(huggingface_hub.utils._errors.RepositoryNotFoundError):
        model = SimilarityLLM("xxx")
