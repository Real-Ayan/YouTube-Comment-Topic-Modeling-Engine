#%%

import pandas as pd
from sentence_transformers import SentenceTransformer

#%%
comments = pd.read_csv("../data/single_video_comments/seed_1.csv")
comments.info()

#%%
comments= comments.reset_index(drop=True)

#%%
texts=comments["CommentText"].astype(str).tolist()

#%%
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
)

#%%
comments["embedding"] = list(embeddings)

#%% 
comments.info()
# %%
comments.to_csv("../data/embedded_dfs/seed_1.csv")
# %%
