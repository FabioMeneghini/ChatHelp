import re
from document import Document

class SAMDocument(Document):
    def __init__(self, path):
        super().__init__(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._text = f.read()
        except Exception as e:
            raise e


    # Dato che i file sorgenti SAM sono composti da più documenti, allora il widget per la selezione del file
    # deve permettere l'inserimento di più file nello stesso formato, che andranno poi uniti.
    # Prima di unirli, verificare che abbiano tutti il formato SAM: per i PDF, invece, se ne può caricare solo uno.

    # 1) unire tutti i documenti in un unico testo;
    # 2) il testo complessivo viene diviso in sezioni (cerca solo tag [H1], [H2] (riga.startswith("[H1") o riga.startswith("[H2"));
    # 3) le sezioni che contengono più parole di max_length vengono ulteriormente suddivise in chunk di max_length parole, con overlap di overlap parole;
    
    def _get_sections(self): # ritorna un generatore di tuple (titolo, testo)
        current_section_title = None
        current_section_text = None
        for line in self._text.split('\n'):
            if line.startswith("[H1") or line.startswith("[H2"):
                title = re.sub(r"\[.*?\]", "", line).strip()
                if current_section_title is not None:
                    yield (current_section_title, current_section_text.strip())
                current_section_title = title
                current_section_text = title + ": \n"
            else:
                if current_section_text is not None:
                    current_section_text += f"{line}\n"
        if current_section_title is not None:
            yield (current_section_title, current_section_text.strip())
