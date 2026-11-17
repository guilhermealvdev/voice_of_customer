from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL, COLLECTION_NAME
)
from prepare_data import load_reviews

def main():
    reviews = load_reviews(DATA_PATH)
    print(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Recria a coleção para evitar duplicação quando a ingestão for executada novamente.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 128
    total = len(reviews)
    for start in range(0, total, batch_size):
        batch = reviews.iloc[start:start + batch_size]
        documents = batch["review_comment_message"].tolist()
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        ids = batch["review_id"].tolist()
        metadatas = [
            {
                "order_id": row.order_id,
                "review_score": int(row.review_score),
                "review_creation_date": str(row.review_creation_date),
            }
            for row in batch.itertuples(index=False)
        ]
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"Indexadas {min(start + batch_size, total)} de {total} avaliações")

    print(f"Ingestão concluída. Total indexado: {collection.count()}")

if __name__ == "__main__":
    main()
