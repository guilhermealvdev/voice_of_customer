from sentence_transformers import SentenceTransformer
import chromadb
import ollama

from config import (
    CHROMA_PATH, EMBEDDING_MODEL, OLLAMA_MODEL, TOP_K,
    MAX_DISTANCE, COLLECTION_NAME
)

class VoiceOfCustomerRAG:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            self.collection = self.client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "Base vetorial não encontrada. Execute primeiro: python src/ingest.py"
            ) from exc

    def retrieve(self, question, top_k=TOP_K):
        query_embedding = self.embedding_model.encode(
            [question], normalize_embeddings=True
        ).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        evidence = []
        for doc, metadata, distance, review_id in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        ):
            if distance <= MAX_DISTANCE:
                evidence.append({
                    "review_id": review_id,
                    "text": doc,
                    "score": metadata.get("review_score"),
                    "date": metadata.get("review_creation_date"),
                    "distance": float(distance),
                })
        return evidence

    def answer(self, question):
        evidence = self.retrieve(question)
        if not evidence:
            return (
                "Não encontrei avaliações suficientemente semelhantes para "
                "responder com segurança. Tente reformular a pergunta ou "
                "perguntar sobre temas presentes nas avaliações."
            ), []

        context_parts = []
        for item in evidence:
            context_parts.append(
                f"[Avaliação {item['review_id']} | nota {item['score']} | "
                f"data {item['date']}]: {item['text']}"
            )
        context = "\n".join(context_parts)

        prompt = f'''
Você é um analista de Voice of Customer de um e-commerce.
Responda em português do Brasil, usando exclusivamente as evidências abaixo.
Não invente números, causas, tendências ou informações que não estejam explícitas.
Se as evidências não permitirem uma conclusão, diga isso claramente.
Resuma padrões apenas quando houver suporte nos comentários recuperados.
Não afirme que a amostra representa todas as avaliações da base.

Pergunta: {question}

Evidências recuperadas:
{context}

Resposta fundamentada:
'''
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"].strip(), evidence
