
import os
import shutil
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from utils import load_all_rule_books, load_yaml_config
from paths import METADATA_FPATH, VECTOR_DB_DIR


def initialize_db(
    persistent_directory: str = VECTOR_DB_DIR,
    collection_name: str = "rule_books",
    delete_existing: bool = False, 
) -> chromadb.Collection:
  
  if os.path.exists(persistent_directory) and delete_existing:
    shutil.rmtree(persistent_directory)

  os.makedirs(persistent_directory, exist_ok=True)

  client = chromadb.PersistentClient(path=persistent_directory)

  collection= client.get_or_create_collection(
    name=collection_name,
    metadata={
              "hnsw:space": "cosine",
              "hnsw:batch_size": 10000,
            })
  
  print(f"ChromaDB initialized with persistent storage at: {persistent_directory}")
  return collection
  

def get_db_collection(
    persistent_directory: str = VECTOR_DB_DIR,
    collection_name: str = "rule_books"
) -> chromadb.Collection:
  return chromadb.PersistentClient(path=persistent_directory).get_collection(name=collection_name)

def chunk_rule_book(rule_book: str, chunk_size: int=1000 , chunk_overlap: int=200) -> list[str]:
  text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
  )
  return text_splitter.split_text(rule_book)

def embed_documents(documents: list[str]) -> list[list[int]]:
  device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
  )
  model= HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-MiniLM-L6-v2",
     model_kwargs={"device": device},
  )
  return model.embed_documents(documents)

def insert_rule_books(collection: chromadb.Collection, rule_books: list[str], metadatas: list[dict[str,any]]):
  next_id=collection.count()
  for i,rule in enumerate(rule_books):
    rule_metadata=metadatas[i]
    chunked_rule=chunk_rule_book(rule)
    embeddings=embed_documents(chunked_rule)
    chunked_metadatas=[rule_metadata]*len(chunked_rule)
    ids= list(range(next_id,next_id+len(chunked_rule)))
    ids=[f"document_{id}" for id in ids]
    collection.add(
      documents=chunked_rule,
      metadatas=chunked_metadatas,
      ids=ids,
      embeddings=embeddings
    )
    next_id+=len(chunked_rule)

def main():
    collection=initialize_db(
      persistent_directory=VECTOR_DB_DIR,
      collection_name="rule_books",
      delete_existing=True
    )
    rule_books=load_all_rule_books()
    metadatas=load_yaml_config(METADATA_FPATH)
    insert_rule_books(collection,rule_books,metadatas)
    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
      main()
