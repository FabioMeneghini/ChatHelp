import sys
import huggingface_hub
import pytest
sys.path.append('src')
from zero_shot_llm import ZeroShotLLM

def test_classify_1():
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    assert zs.classify("La richiesta non ha nulla a che fare con il contesto fornito.") == "impossibile"

def test_classify_2():
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    assert zs.classify("La richiesta è pertinente con il contesto fornito.") == "pertinente"

def test_exception():
    with pytest.raises(OSError):
        zs = ZeroShotLLM("xxx", "impossibile", "pertinente")