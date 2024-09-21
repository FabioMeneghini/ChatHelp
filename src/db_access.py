import psycopg2
from document import Document
from similarity_llm import SimilarityLLM

class DBAccess:
    def __init__(self, host, database, user, password, port):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None
        self.outcome = self.connect()

    def connect(self):
        if self.connection:
            return "Connessione già aperta."
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            return "Connessione al database riuscita."
        except (psycopg2.OperationalError, psycopg2.Error) as error:
            return f"Errore di connessione al database: {error}"
    
    def get_sections_by_code(self, codes):
        if not self.connection:
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT sezione FROM docs WHERE codice IN %s;", (tuple(codes),))
            result = cursor.fetchall()
            return result
        except (Exception, psycopg2.Error) as error:
            return f"Errore nel recupero delle sezioni: {error}"
        finally:
            cursor.close()
    
    def close_connection(self):
        if self.connection:
            self.connection.close()
            self.outcome = "Connessione al database chiusa."
            self.connection = None
    
    def get_bm25_rank(self, keywords: str):
        if not self.connection:
            return None
        keywords = keywords.replace(" ", " | ") # sostituisce gli spazi con l'operatore OR
        safe_input = psycopg2.extensions.adapt(keywords).getquoted().decode('latin-1') # sanifica input
        try:
            cursor = self.connection.cursor() # crea un cursore per eseguire le query
            BM25_RANK_QUERY = f"""
                with query as (
                    select to_tsquery({safe_input}) as qq
                ),
                adoc as (
                    select avg(array_length(tsvector_to_array(to_tsvector(testo)), 1)) as avglen from docs
                )
                select codice, cast(bm25rank as float), (select testo from docs where t5.codice=docs.codice), (select sezione from docs where t5.codice=docs.codice) from (
                    select codice, sum(ff) as bm25rank from (
                        select codice, t3.lexeme, f,doclen, (select avglen from adoc) as avglen, idf.idf, idf.idf*f*2.5/(f+1.5*(1-0.75+0.75*doclen/(select avglen from adoc))) as ff from (
                            select codice, (t2.t).lexeme as lexeme, array_length((t2.t).positions, 1) as f, doclen from (
                                select codice, unnest(t) as t, array_length(tsvector_to_array(testo), 1) as doclen from (
                                    select codice, to_tsvector(testo) as testo, ts_delete(
                                        to_tsvector(testo),
                                        tsvector_to_array(ts_delete(to_tsvector(testo), array(
                                            select unnest(string_to_array(replace(qq::text, '''', ''), ' | ')))))) AS t
                                    from docs, query where to_tsvector(testo) @@ (select qq from query)
                                ) t1
                            ) t2 where (t2.t).lexeme @@ (select qq from query)
                        ) t3 left outer join idf on t3.lexeme=idf.lexeme
                    ) t4 group by codice order by 2 desc
                ) t5 LIMIT 20;
            """
            cursor.execute(BM25_RANK_QUERY)
            result = cursor.fetchall() # restituisce il risultato della query
            return result
        except (Exception, psycopg2.Error) as error:
            return f"Errore nel calcolo del rank BM25: {error}"
        finally: # chiude il cursore
            cursor.close()
    
    def get_embeddings_rank(self, input: str, similarity_model: SimilarityLLM):
        if not self.connection:
            return None
        safe_input = psycopg2.extensions.adapt(input).getquoted().decode('latin-1') # sanifica input
        vector = similarity_model.get_vector(safe_input) # calcola l'embedding del testo di input
        try:
            cursor = self.connection.cursor()
            EMBEDDIGS_RANK_QUERY = f"SELECT codice, vettore <=> '{vector}' as score, testo, sezione FROM docs ORDER BY score LIMIT 20;" # calcola la distanza tra l'embedding del testo di input e gli embeddings dei documenti con cosine similarity
            cursor.execute(EMBEDDIGS_RANK_QUERY)
            result = cursor.fetchall()
            return result
        except (Exception, psycopg2.Error) as error:
            return f"Errore nel calcolo del rank con embeddings: {error}"
        finally:
            cursor.close()
    
    def calculate_idf(self): # (ri)calcola l'IDF: utile per quando vengono aggiunti documenti al database oppure viene caricato un nuovo file
        if not self.connection:
            yield "Non connesso al database."
            return
        try:
            yield "Sto calcolando gli IDF...\n\n"
            cursor = self.connection.cursor()
            CALCULATE_IDF_QUERY = """
                DROP TABLE IF EXISTS idf;
                CREATE TABLE idf AS
                SELECT lexeme, cnt, n, ln((n-cnt+0.5)/(cnt+0.5)+1) AS idf FROM (
                    SELECT lexeme, count(*) AS cnt, n FROM (
                        SELECT (t2.t).lexeme AS lexeme, n AS n FROM (
                            SELECT unnest(t) AS t, n AS n FROM (
                                SELECT to_tsvector(testo) AS t, count(*) OVER() AS n FROM docs
                            ) t1
                        ) t2
                    ) t3 GROUP BY lexeme, n
                ) t4;
            """
            cursor.execute(CALCULATE_IDF_QUERY)
            self.connection.commit() # conferma le modifiche al database
            yield "IDF ricalcolati e salvati con successo."
        except (Exception, psycopg2.Error) as error:
            yield f"Errore durante il calcolo degli IDF: {error}"
        finally:
            cursor.close()

    def calculate_embeddings(self, similarity_model: SimilarityLLM): # (ri)calcola gli embeddings: utile se viene cambiato il modello per il calcolo degli embedding
        if not self.connection:
            yield "Non connesso al database."
            return
        try:
            yield "Sto calcolando gli embeddings...\n\n"
            cursor = self.connection.cursor()
            cursor.execute("select codice, testo from docs;")
            result = cursor.fetchall()
            for row in result:
                vector = similarity_model.get_vector(row[1])
                cursor.execute("update docs set vettore = %s where codice = %s;", (vector, row[0]))
            self.connection.commit()
            yield "Embeddings ricalcolati e salvati con successo."
        except (Exception, psycopg2.Error) as error:
            yield f"Errore durante il calcolo degli embeddings: {error}"
        finally:
            cursor.close()
    
    def _overwrite_sections(self, sections): # inserisce nuovi documenti nel database
        if not self.connection:
            return "Non connesso al database."
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM docs;")
            i = 0
            for section in sections:
                cursor.execute("INSERT INTO docs (codice, testo, sezione) VALUES (%s, %s, %s);", (i, section[0], section[1]))
                i += 1
            self.connection.commit()
            return "Documento inserito con successo."
        except (Exception, psycopg2.Error) as error:
            return f"Errore durante l'inserimento del documento: {error}"
        finally:
            cursor.close()
    
    def insert_new_document(self, document: Document):
        if not self.connection:
            yield "Non connesso al database."
            return
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM file; INSERT INTO file (nome) VALUES (%s);", (document.get_name(),))
            self.connection.commit()
            sections = list(document.get_docs())
            yield "Sto inserendo il nuovo documento...\n\n"
            out = self._overwrite_sections(sections)
            yield out
        except (Exception, psycopg2.Error) as error:
            yield f"Errore durante l'inserimento del documento: {error}"
        finally:
            cursor.close()
    
    def get_all_rows(self):
        if not self.connection:
            return "Non connesso al database."
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT codice, testo, sezione FROM docs;")
            result = cursor.fetchall()
            return result
        except (Exception, psycopg2.Error) as error:
            return f"Errore nel recupero dei documenti: {error}"
        finally:
            cursor.close()
    
    def get_document_name(self):
        if not self.connection:
            return "Non connesso al database."
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT nome FROM file;")
            result = cursor.fetchone()
            if not result:
                return "Nessun documento caricato."
            return result[0]
        except (Exception, psycopg2.Error) as error:
            return f"Errore durante il recupero del nome del documento: {error}"
        finally:
            cursor.close()
    
"""
    def _insert_vector(self, vector, code):
        if not self.connection:
            return "Non connesso al database."
        try:
            cursor = self.connection.cursor()
            cursor.execute("UPDATE docs SET vettore = %s WHERE codice = %s;", (vector, code))
            self.connection.commit()
            return "Vettore inserito con successo."
        except (Exception, psycopg2.Error) as error:
            return f"Errore durante l'inserimento del vettore: {error}"
        finally:
            cursor.close()
"""