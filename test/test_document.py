import sys
import pytest
sys.path.append('src')
from document import Document
from pdf_document import PDFDocument
from sam_document import SAMDocument

def test_get_path():
    doc = Document('test/file/test1.pdf')
    assert doc.get_path() == 'test/file/test1.pdf'

def test_get_name():
    doc = Document('test\\file\\test1.pdf')
    assert doc.get_name() == 'test1.pdf'

def test_get_name_error():
    with pytest.raises(FileNotFoundError):
        doc = Document('test/file/xxx.pdf')

def test_pdf_extractor_1(): #testa se il numero di documenti estratti è corretto
    pdf = PDFDocument('test\\file\\test1.pdf')
    docs = list(pdf.get_docs())
    assert len(docs) == 5

def test_pdf_extractor_2(): #testa se i documenti estratti sono corretti
    pdf = PDFDocument('test/file/test1.pdf')
    docs = list(pdf.get_docs())
    assert docs == [('Intestazione 1: \nContenuto della sezione con titolo “Intestazione 1”.', 'Intestazione 1'),
                    ('Intestazione 2: \nContenuto della sezione con titolo “Intestazione 2”.', 'Intestazione 2'),
                    ('Intestazione 3: \nContenuto della sezione con titolo “Intestazione 3”.', 'Intestazione 3'),
                    ('Intestazione 4: \nContenuto della sezione con titolo “Intestazione 4”.', 'Intestazione 4'),
                    ('Intestazione 5: \nContenuto della sezione con titolo “Intestazione 5”.', 'Intestazione 5')]

def test_pdf_extractor_3(): #testa se i documenti estratti hanno dimensione corretta (in parole)
    pdf = PDFDocument('test/file/test2docCRM.pdf')
    docs = list(pdf.get_docs(290, 50))
    docs = [doc[0] for doc in docs]
    ok = True
    for doc in docs:
        assert len(doc.split()) <= 340 #290 + 50
    assert ok

# def test_pdf_extractor_4(): #nome file inesistente
#     with pytest.raises(pymupdf.FileNotFoundError):
#         pdf = PDFDocument('test/file/xxx.pdf')

def test_sam_extractor_1(): #testa se il numero di documenti estratti è corretto
    sam = SAMDocument('test\\file\\test3.sam')
    docs = list(sam.get_docs())
    assert len(docs) == 4

def test_sam_extractor_2(): #testa se i documenti estratti sono corretti
    sam = SAMDocument('test/file/test3.sam')
    docs = list(sam.get_docs())
    print(docs)
    assert docs == [('Intestazione 1: \nContenuto della sezione con titolo "Intestazione 1".', 'Intestazione 1'),
                    ('Intestazione 2: \nContenuto della sezione con titolo "Intestazione 2".', 'Intestazione 2'),
                    ('Intestazione 3: \nContenuto della sezione con titolo "Intestazione 3".', 'Intestazione 3'),
                    ('Intestazione 4: \nContenuto della sezione con titolo "Intestazione 4".', 'Intestazione 4')]

def test_sam_extractor_3(): #testa se i documenti estratti hanno dimensione corretta (in parole)
    sam = SAMDocument('test/file/test4docSupersam.sam')
    docs = list(sam.get_docs(290, 50))
    docs = [doc[0] for doc in docs]
    ok = True
    for doc in docs:
        assert len(doc.split()) <= 340 #290 + 50
    assert ok
