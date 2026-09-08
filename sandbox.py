import pandas as pd
from sentence_transformers import SentenceTransformer
from chunking import chunk_text
from index_to_qdrant import build_points_for_game

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
tok = model.tokenizer

df = pd.read_parquet("data/games_for_rag.parquet")
df = df.dropna(subset=["embedding_text"])
sample = df.sample(n=500, random_state=42)

MAX_TOKENS = 450
worst = None

for _, row in sample.iterrows():
    points = build_points_for_game(row, tok, MAX_TOKENS)
    for point in points:
        text = point["text"]
        n_real = len(tok(text, truncation=False)["input_ids"])  # compte réel, avec tokens spéciaux
        if n_real > 512:
            gap = n_real - len(tok.tokenize(text))
            if worst is None or n_real > worst[2]:
                worst = (row["name"], point["id"], n_real, gap)

if worst:
    print(f"COUPABLE : {worst[0]} (point id {worst[1]})")
    print(f"  Compte réel (encode, tokens spéciaux inclus) : {worst[2]}")
    print(f"  Écart avec notre comptage (tokenize) : {worst[3]}")
else:
    print("Aucun chunk individuel ne dépasse 512 tokens réels.")