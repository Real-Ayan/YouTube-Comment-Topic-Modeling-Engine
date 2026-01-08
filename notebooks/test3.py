#%% 
import pandas as pd
import numpy as np

import umap
import hdbscan

#%%
comments=pd.read_csv("../data/embedded_dfs/seed_1.csv")
# %%
comments.info()


#%%
embeddings=comments["embedding"].copy()
embeddings.info()

#%%
type(embeddings[0])
# type(embeddings["embedding"].iloc[0])

#%%


X = np.vstack(
    embeddings
    .apply(lambda s: np.fromstring(s.strip("[]"), sep=" "))
    .values
).astype("float32")

#%%
X.shape


#%%

umap_model =umap.UMAP(
    n_neighbors=15,
    n_components=30,
    metric="cosine",
    random_state=42
)

embeddings_umap = umap_model.fit_transform(X)

#%%
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=3,
    metric='euclidean',
    cluster_selection_method="eom"
)

#%%
labels= clusterer.fit_predict(embeddings_umap)

#%%
comments["cluster_id"] = labels
comments["cluster_id"].value_counts()

#%%
clustered_df = comments[comments["cluster_id"] != -1]

#%%
cluster_sizes = clustered_df["cluster_id"].value_counts()
valid_clusters = cluster_sizes[cluster_sizes >= 5].index

clustered_df = clustered_df[clustered_df["cluster_id"].isin(valid_clusters)]

#%%
top_comments = (
    clustered_df
    .sort_values(["cluster_id", "Likes"], ascending=[True, False])
    .groupby("cluster_id")
    .head(10)
)

# %%
top_comments.head(40)
# %%
