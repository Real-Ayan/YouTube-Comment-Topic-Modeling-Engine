## YouTube Comment Topic Modeling Engine

This repository provides an unsupervised machine learning pipeline to replicate YouTube’s "Topic Summarization" feature. By grouping thousands of comments into semantically similar "Opinion Bubbles," the engine transforms flat comment sections into structured, navigable insights.

#### Key Features
* **Semantic Clustering:** Uses Transformer-based embeddings to group comments by intent and topic rather than just keyword matching.
* **Dynamic Parameter Scaling:**Implements a heuristic-based approach ($min\_cluster\_size = \sqrt{N}$) to adapt to varying comment volumes.
* **Noise Reduction:** Leverages HDBSCAN to automatically filter out "troll" comments or spam that do not belong to a coherent discussion theme.

### Data

The main dataset is - 
["YouTube Comments Sentiment Dataset" from Kaggle](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset)



### Methodology 

The pipeline follows a state-of-the-art BERTopic-style architecture for dense vector clustering:

1. **Embedding:** all-MiniLM-L6-v2 (Sentence-Transformers) maps comments to a 384-dimensional dense vector space.

2. **Dimensionality Reduction:** **UMAP** reduces vectors to a lower-dimensional manifold (20 components) while preserving local topological structure.

3. **Clustering:** **HDBSCAN** performs density-based clustering to identify natural groupings of varying shapes and densities.

```
 embedding model("all-MiniLM-L6-v2")
            |
            v
 Dimensinality Reduction(UAMP) 
            |
            v
 Clustering(HDBSCAN)
            |
            v
 Filtering(optional)
            |
            v
 Topic Segregation
```
### Technical Stack

- **Language:** Python 3.x

- **Libraries:** `pandas`, `umap-learn`, `hdbscan`, `sentence-transformers`, `scikit-learn`

