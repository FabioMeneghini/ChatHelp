from groq import Groq
import os
from dotenv import load_dotenv

class QuestionAnsweringLLM:
    names = ["mixtral-8x7b-32768", "llama3-8b-8192", "gemma2-9b-it"]

    def __init__(self, model_name):
        if model_name not in QuestionAnsweringLLM.names:
            raise RuntimeError("Modello non valido.")
        else:
            load_dotenv()
            key = os.getenv("GROQ_API_KEY") # recupera la chiave API da una variabile d'ambiente
            self.client = Groq(api_key=key) # si collega al servizio Groq usando la chiave API
            self.model_name = model_name
    
    def get_name(self):
        return self.model_name
    
    def answer(self, question, context):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """Sei un assistente virtuale esperto che risponde a domande in italiano.
                                    Di seguito di verrà fornita una domanda dall'utente e un contesto, e riguarderanno la documentazione relativa ad un software.
                                    Rispondi alla domanda basandoti esclusivamente sul contesto fornito,
                                    dando una spiegazione dettagliata ed esaustiva della risposta data.
                                    Se possibile rispondi con un elenco puntato o numerato.
                                    Se la domanda non ha nulla a che fare con il contesto, la tua risposta deve essere esattamente la seguente: "Mi dispiace, ma non sono in grado di rispondere a questa domanda"."""
                    },
                    {
                        "role": "user",
                        "content": f"""DOMANDA: {question}\n\nCONTESTO: {context}""",
                    }
                ],
                model=self.model_name # specifica il modello da utilizzare
            )
            return chat_completion.choices[0].message.content # restituisce la risposta alla domanda
        except Exception as e:
            return f"Si è verificato un errore:\n {e}\n\nRiprova più tardi con lo stesso modello oppure riprova con un modello diverso."
    
    def check(self, question):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{question}",
                    }
                ],
                model="llama-guard-3-8b",
            )
            return chat_completion.choices[0].message.content # 'safe' se la domanda è sicura, 'unsafe' se la domanda è pericolosa, seguita dai contenuti pericolosi
        except Exception as e:
            return f"Si è verificato un errore:\n {e}\n\nRiprova più tardi con lo stesso modello oppure riprova con un modello diverso."