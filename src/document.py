import os

class Document:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Il percorso {path} non esiste.")
        self._path = path

    def get_path(self):
        return self._path

    def get_name(self):
        return self._path.split("\\")[-1]

    def _get_sections(self):
        raise NotImplementedError("Questo metodo deve essere implementato nelle sottoclassi")
    
    def _chunk_text(self, text, max_length=290, overlap=50):
        words = text.split()
        chunks = []
        step = max_length - overlap
        for i in range(0, len(words), step):
            start = max(0, i - overlap)
            end = min(len(words), i + max_length)
            chunk = words[start:end]
            chunks.append(' '.join(chunk))
        return chunks

    def get_docs(self, max_length=290, overlap=50):
        for section in self._get_sections():
            if len(section[1].split()) > max_length:
                chunks = self._chunk_text(section[1], max_length, overlap)
                for chunk in chunks:
                    yield (chunk, section[0])
            else:
                yield (section[1], section[0])
