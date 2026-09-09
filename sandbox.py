import pandas as pd
df = pd.read_parquet("data/games_for_rag.parquet")


row = df[df["name"] == "Ninnin"]
print(row[["price", "tags"]])