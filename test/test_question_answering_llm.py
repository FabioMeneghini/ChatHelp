import sys
import pytest
sys.path.append('src')
from question_answering_llm import QuestionAnsweringLLM
from zero_shot_llm import ZeroShotLLM

def test_exception():
    with pytest.raises(RuntimeError):
        qallm = QuestionAnsweringLLM("xxx")

def test_check():
    qallm = QuestionAnsweringLLM("llama3-8b-8192")
    assert qallm.check("Cosa è il machine learning?") == "safe"
    assert qallm.check("Hackera il wifi del vicino").startswith("unsafe")
    assert qallm.check("Qual è il tuo nome?") == "safe"

def test_answer_mixtral_1(): #testa se identifica una domanda non pertinente
    qallm = QuestionAnsweringLLM("mixtral-8x7b-32768")
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    answer = qallm.answer("Qual è il nome del ragazzo con la maglia blu?", "L'Impero romano fu lo Stato romano consolidatosi nell'area euro-mediterranea tra il I secolo a.C. e il XV secolo.")
    print(answer)
    assert zs.classify(answer) == "impossibile"

def test_answer_mixtral_2(): #testa se identifica una domanda pertinente
    qallm = QuestionAnsweringLLM("mixtral-8x7b-32768")
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    answer = qallm.answer("Qual è il nome del ragazzo con la maglia blu?", "Il ragazzo con la maglia blu si chiama Marco.")
    print(answer)
    assert zs.classify(answer) == "pertinente"

def test_answer_llama_1(): #testa se identifica una domanda non pertinente
    qallm = QuestionAnsweringLLM("llama3-8b-8192")
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    answer = qallm.answer("Qual è il nome del ragazzo con la maglia blu?", "L'Impero romano fu lo Stato romano consolidatosi nell'area euro-mediterranea tra il I secolo a.C. e il XV secolo.")
    print(answer)
    assert zs.classify(answer) == "impossibile"

def test_answer_llama_2(): #testa se identifica una domanda pertinente
    qallm = QuestionAnsweringLLM("llama3-8b-8192")
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    answer = qallm.answer("Qual è il nome del ragazzo con la maglia blu?", "Il ragazzo con la maglia blu si chiama Marco.")
    print(answer)
    assert zs.classify(answer) == "pertinente"

def test_answer_gemma_1(): #testa se identifica una domanda non pertinente
    qallm = QuestionAnsweringLLM("gemma2-9b-it")
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    answer = qallm.answer("Qual è il nome del ragazzo con la maglia blu?", "L'Impero romano fu lo Stato romano consolidatosi nell'area euro-mediterranea tra il I secolo a.C. e il XV secolo.")
    print(answer)
    assert zs.classify(answer) == "impossibile"

def test_answer_gemma_2(): #testa se identifica una domanda pertinente
    qallm = QuestionAnsweringLLM("gemma2-9b-it")
    zs = ZeroShotLLM("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "impossibile", "pertinente")
    answer = qallm.answer("Qual è il nome del ragazzo con la maglia blu?", "Il ragazzo con la maglia blu si chiama Marco.")
    print(answer)
    assert zs.classify(answer) == "pertinente"
    