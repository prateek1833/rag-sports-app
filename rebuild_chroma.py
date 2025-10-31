# rebuild_chroma.py
import shutil, time
from pathlib import Path
from langchain.document_loaders import TextLoader
from utils.rag_utils import get_embedder, create_chroma_db

# load documents (safe)
docs = []
for i in range(1, 11):
    p = Path(f"./documents/{i}.txt")
    if p.exists():
        docs.extend(TextLoader(str(p)).load())
    else:
        print("Missing:", p)

# choose the embedder model you want the DB to use
# If you want 768-dim (existing DB), use all-mpnet-base-v2
# If you want 384-dim, use all-MiniLM-L6-v2
embed_model = "sentence-transformers/all-mpnet-base-v2"  # -> 768 dims
# embed_model = "sentence-transformers/all-MiniLM-L6-v2"  # -> 384 dims

embedder = get_embedder(model_name=embed_model)

# delete existing DB (be sure it's not open)
p = Path("./chroma_db")
if p.exists():
    print("Removing existing chroma_db ...")
    shutil.rmtree(p)

# create DB
db = create_chroma_db(docs, persist_dir=str(p), embedder=embedder)
print("Created chroma_db with embed model:", embed_model)
