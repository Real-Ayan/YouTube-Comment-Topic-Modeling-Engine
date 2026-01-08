#%%
import os
import pandas as pd
from dotenv import load_dotenv

from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from langchain.embeddings.base import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

from pymongo import MongoClient 
from pymongo.database import Database
from pymongo.collection import Collection



#%%
load_dotenv()
class Settings:
    ATLAS_URI=os.environ.get("ATLAS_URI")
    MODEL_NAME="all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS=384
    DB_NAME="Opinion_trend"
    COLLECTION_NAME="comments_data"
    VECTOR_INDEX="comments_vectors"

settings = Settings()



#%%
# class Comments(BaseModel):
#     commentID

#%%
class LocalSentenceTransformerEmbedding(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model= SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()
    
    def embed_query(self, text):
        return self.model.encode(
            text,
            normalize_embeddings=True

        ).tolist()
    

#%%
def ensure_collection_exists(db_client:Database, collection:Collection)-> None:
    if collection not in db_client.list_collection_names():
        print(f"Warn: Collection : {collection} not found. Creating it now....")
        db_client.create_collection(collection)
        print(f"Info: Collection '{collection}' created successfully .")

#%%
def ensure_vector_index_exists(collection:Collection,vector_index: str):
    existing_indexes ={ idx['name'] for idx in collection.list_search_indexes()}
    if vector_index not in existing_indexes:
        print(f"Info: vector index: '{vector_index}' not present in collection '{collection}' ")
        print(f"Info: creating vector index {vector_index} on collection {collection}...")
        index_defination={
            "name": vector_index,
            "type": "vectorSearch",
            "definition":{
                "fields":[
                    {
                        "type":"vector",
                        "path":"embeddings",
                        "numDimensions": settings.EMBEDDING_DIMENSIONS,
                        "similarity":"dotProduct"
                    },
                ]
            }
        }    
        collection.create_search_index(model=index_defination)

#%%
def prepare_documents(comments:pd.DataFrame):
    documents: list[Document] = []
    if comments.empty:
        print(f"Warn: No comments")
        return documents
    documents=[
        Document(
            page_content = row["CommentText"],
            metadata={
                "comment_id":row["CommentID"],
                "Likes":int(row["Likes"]),
                "replies":int(row["Replies"]),
                "video_title":row["VideoTitle"]
            }
        )
        for _, row in comments.iterrows()
    ]
    return documents



#%%
client = MongoClient(host=settings.ATLAS_URI,tz_aware=True)
client.admin.command('ping')
db=client[settings.DB_NAME]

ensure_collection_exists(db,settings.COLLECTION_NAME)
collection:Collection = db[settings.COLLECTION_NAME]

ensure_vector_index_exists(collection,settings.VECTOR_INDEX)


#%%
embedding = LocalSentenceTransformerEmbedding()

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding,
    dimensions=settings.EMBEDDING_DIMENSIONS,
    index_name=settings.VECTOR_INDEX,
    relevance_score_fn="dotProduct",
)

#%%
data = pd.read_csv("../data/single_video_comments/seed_1.csv")
data.info()
#%%
documents=prepare_documents(data)
#%%
vector_store.add_documents(documents)
# %%


#%%
