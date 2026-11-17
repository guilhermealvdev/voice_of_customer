import pandas as pd
from config import DATA_PATH

REQUIRED_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_comment_message",
    "review_creation_date",
]

def load_reviews(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Baixe o dataset Brazilian E-Commerce Public Dataset by Olist "
            "no Kaggle e coloque o arquivo olist_order_reviews_dataset.csv em data/raw/."
        )

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["review_comment_message"] = df["review_comment_message"].fillna("").astype(str).str.strip()
    df["review_id"] = df["review_id"].fillna("").astype(str)
    df["order_id"] = df["order_id"].fillna("").astype(str)
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["review_creation_date"] = df["review_creation_date"].fillna("").astype(str)

    # Comentários vazios não contêm texto para indexação semântica.
    df = df[
        (df["review_comment_message"] != "")
        & (df["review_id"] != "")
        & (df["review_score"].notna())
    ].drop_duplicates(subset=["review_id"])

    df["review_score"] = df["review_score"].astype(int)
    return df.reset_index(drop=True)

if __name__ == "__main__":
    reviews = load_reviews()
    print(f"Avaliações textuais válidas para a base: {len(reviews)}")
    print(reviews.head(3).to_string(index=False))
