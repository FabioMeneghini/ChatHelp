import fitz  # PyMuPDF
from document import Document

class PDFDocument(Document):
    def __init__(self, path):
        super().__init__(path)
        try:
            self._document = fitz.open(path)
        except Exception as e:
            raise e
    
    def _is_bold(self, span):
        return span['flags'] & 2**4  # il flag 4 indica il grassetto

    def _is_header(self, span):
        return span['size'] > 12

    def _get_section(self, bold_text, text):
        return f"{bold_text}: \n{' '.join(text)}"

    def _get_sections(self):
        current_bold_text = None
        current_text = []
        for page_number in range(len(self._document)):
            page = self._document.load_page(page_number)
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['bbox'][1] < page.rect.height * 0.1 or block['bbox'][1] > page.rect.height * 0.9:
                    continue
                if 'lines' in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if self._is_bold(span) and self._is_header(span):
                                if current_bold_text is not None: # restituisce il blocco di testo precedente se esiste
                                    yield (current_bold_text, self._get_section(current_bold_text, current_text))
                                # aggiorna il testo in grassetto corrente
                                current_bold_text = span["text"]
                                current_text = []
                            else:
                                if current_bold_text:
                                    current_text.append(span["text"])
        if current_bold_text: # restituisce l'ultimo blocco di testo se esiste
            yield (current_bold_text, self._get_section(current_bold_text, current_text))
