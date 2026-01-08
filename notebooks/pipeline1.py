#%%
import pandas as pd
import math
import numpy as np
import umap
import hdbscan

from sentence_transformers import SentenceTransformer as ST
from sklearn.feature_extraction.text import TfidfVectorizer

file_name="seed_1"



#%%
#DATA ENTRY POINT HERE
comments = pd.read_csv(f"../data/raw_dfs/{file_name}_raw.csv")

#%%
total_comments = len(comments)
dynamic_min_size = max(5, int(math.sqrt(total_comments)))
dynamic_min_samples = max(5, int(dynamic_min_size * 0.15))

#%%
EMBEDDING_MODEL="all-MiniLM-L6-v2"
embedding_model = ST(EMBEDDING_MODEL)

#%%
# Dimensionality reduction from (all-MiniLM-L6-v2)384 to lower like 40 or something 
# Thinking of using HDBSCAN , So I am trying to keep the dimention under 50
umap_model= umap.UMAP(
    n_neighbors=10,
    n_components=5,
    metric="cosine",
    random_state=42,
)

#%%
# Using HDBSCAN as number of clusters is not known
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=dynamic_min_size,
    min_samples=dynamic_min_samples,
    metric='euclidean',
    cluster_selection_method="eom"
)

#%%
comments.info()

#%%
comments = comments.reset_index(drop=True)

#%%
comment_texts= comments["CommentText"].astype(str).tolist()

embeddings = embedding_model.encode(
    comment_texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)

#%%
embeddings_umap = umap_model.fit_transform(embeddings)

#%%
labels= clusterer.fit_predict(embeddings_umap)

#%%
comments["cluster_id"] = labels

comments.to_csv(f"../data/clustered_dfs/{file_name}_clustered_run_1.csv")

#%%
def extract_topic_keywords(df, cluster_col='cluster_id', text_col='CommentText'):
    # 1. Group all text by cluster
    docs_per_class = df.groupby([cluster_col], as_index=False).agg({text_col: ' '.join})
    
    # 2. Apply c-TF-IDF
    # We use standard TF-IDF, but treat each CLUSTER as a single document
    count = TfidfVectorizer(stop_words="english", max_features=10)
    c_tf_idf = count.fit_transform(docs_per_class[text_col])
    words = count.get_feature_names_out()
    
    # 3. Extract top words per cluster
    topic_keywords = {}
    for i in range(len(docs_per_class)):
        cluster_label = docs_per_class.iloc[i][cluster_col]
        if cluster_label == -1: continue # Skip noise
        
        # Get top 5 words for this cluster
        indices = c_tf_idf[i].toarray().flatten().argsort()[-5:][::-1]
        topic_keywords[cluster_label] = [words[idx] for idx in indices]
        
    return topic_keywords

keywords = extract_topic_keywords(comments)

#%%
keywords

#%%
