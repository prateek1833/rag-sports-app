# utils/rag_utils.py
import os
import shutil
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Set your desired embedding model here
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  

def get_embedder(model_name=EMBEDDING_MODEL):
    if not model_name:
        raise ValueError("Embedder model_name cannot be None")
    return HuggingFaceEmbeddings(model_name=model_name)

def create_chroma_db(documents, persist_dir="./chroma_db", embedder=None):
    if embedder is None:
        embedder = get_embedder()

    # Check if existing collection exists
    if os.path.exists(persist_dir):
        try:
            db = Chroma(persist_directory=persist_dir, embedding_function=embedder)
            # Check dimensionality
            if db._collection.info()['embedding_dim'] != embedder.embed_query("test").shape[0]:
                print("Embedding dimension mismatch. Recreating collection...")
                shutil.rmtree(persist_dir)
                db = Chroma.from_documents(documents, embedding=embedder, persist_directory=persist_dir)
        except Exception as e:
            print("Error loading collection. Recreating:", e)
            shutil.rmtree(persist_dir)
            db = Chroma.from_documents(documents, embedding=embedder, persist_directory=persist_dir)
    else:
        db = Chroma.from_documents(documents, embedding=embedder, persist_directory=persist_dir)

    db.persist()
    return db

# add this to utils/rag_utils.py (near other helpers)

import os
from typing import Optional

def load_chroma(persist_dir: str = None, embedder=None):
    """
    Load an existing Chroma DB from disk and return the DB object.

    - persist_dir: path to chroma persist directory (default from CHROMA_PERSIST_DIR)
    - embedder: a HuggingFaceEmbeddings (or compatible) instance. If None, will try to create one
                using get_embedder() defined in this file.

    Raises:
        FileNotFoundError if persist_dir doesn't exist
        RuntimeError for other Chroma init problems (with helpful message)
    """
    persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Chroma persist directory not found at '{persist_dir}'. Build the DB first.")

    # ensure embedder available
    if embedder is None:
        try:
            embedder = get_embedder()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedder while loading Chroma: {e}") from e

    # instantiate Chroma with defensive API compatibility across versions
    try:
        # preferred newer signature: Chroma(persist_directory=..., embedding_function=embedder)
        return Chroma(persist_directory=str(persist_dir), embedding_function=embedder)
    except TypeError:
        try:
            # older signature: Chroma(persist_directory=..., embedding=embedder)
            return Chroma(persist_directory=str(persist_dir), embedding=embedder)
        except Exception as e_inner:
            raise RuntimeError(f"Failed to load Chroma DB from '{persist_dir}': {e_inner}") from e_inner
    except Exception as e:
        raise RuntimeError(f"Failed to load Chroma DB from '{persist_dir}': {e}") from e

# add near other functions in utils/rag_utils.py
import shutil
import time
from pathlib import Path

def ensure_chroma_matches_embedder(docs, embedder, persist_dir: str = "./chroma_db", max_retries: int = 3):
    """
    Ensure chroma_db exists and its embedding_dim matches embedder's output dim.
    If mismatch or missing, recreate the DB using create_chroma_db(..., embedder=embedder).
    Returns the Chroma DB instance.
    """
    persist_path = Path(persist_dir)
    # get embedder dim
    try:
        sample_vec = None
        if hasattr(embedder, "embed_query"):
            sample_vec = embedder.embed_query("test")
        elif hasattr(embedder, "embed_documents"):
            sample_vec = embedder.embed_documents(["test"])[0]
        elif hasattr(embedder, "embed"):
            sample_vec = embedder.embed("test")
        else:
            raise RuntimeError("Embedder does not expose embed_query/embed_documents/embed")
        emb_dim = len(sample_vec)
    except Exception as e:
        raise RuntimeError(f"Failed to produce sample embedding: {e}") from e

    # If DB missing -> create
    if not persist_path.exists():
        return create_chroma_db(docs, persist_dir=str(persist_path), embedder=embedder)

    # Try loading DB and read internal dimension
    try:
        try:
            db = Chroma(persist_directory=str(persist_path), embedding_function=embedder)
        except TypeError:
            db = Chroma(persist_directory=str(persist_path), embedding=embedder)
        # collection info
        collection_dim = None
        try:
            info = db._collection.info()
            collection_dim = int(info.get("embedding_dim")) if info and "embedding_dim" in info else None
        except Exception:
            collection_dim = None

        # if cannot determine dims, assume recreate
        if collection_dim is None or collection_dim != emb_dim:
            # remove and recreate safely with retries
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(persist_path)
                    break
                except Exception as ex:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    raise RuntimeError(f"Failed to remove existing chroma_db at {persist_path}: {ex}") from ex
            return create_chroma_db(docs, persist_dir=str(persist_path), embedder=embedder)
        # dims match: return loaded db
        return db
    except Exception:
        # fallback: attempt removal+recreate
        try:
            shutil.rmtree(persist_path)
        except Exception:
            pass
        return create_chroma_db(docs, persist_dir=str(persist_path), embedder=embedder)
