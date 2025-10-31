# main.py (Simplified LLM Judge for F1 Score, Fixed Memory Save, Removed Token Limit)
import os
import json
import re
import torch
from transformers import pipeline, AutoTokenizer
from utils.eval_utils import evaluate_response_detailed
from mcp_client import mcp_list_tools_sync, mcp_call_tool_sync
from utils.rag_utils import get_embedder, create_chroma_db, load_chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# ---------- Load documents safely ----------
docs = []
for i in range(1, 11):
    path = f"./documents/{i}.txt"
    if os.path.exists(path):
        try:
            docs.extend(TextLoader(path).load())
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    else:
        print(f"File not found: {path}")

# ---------- Function to get Chroma DB ----------
def get_db(embed_model_name="BAAI/bge-base-en-v1.5", chunk_size=1000, chunk_overlap=200):
    embedder = get_embedder(model_name=embed_model_name)
    persist_dir = "./chroma_db"

    if os.path.exists(persist_dir):
        try:
            db = load_chroma(embedder=embedder)
            print("Chroma DB loaded from disk.")
        except Exception as e:
            print(f"Failed to load existing DB: {e}. Recreating...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            split_docs = splitter.split_documents(docs)
            db = create_chroma_db(split_docs, embedder=embedder)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(docs)
        db = create_chroma_db(split_docs, embedder=embedder)

    return db, embedder

# --- Simplified LLM Judge (F1 Score Only) ---
def simple_llm_judge(generator, query, sources, answer):
    """Inline LLM Judge: Simplified to compute only F1 score with strict JSON output."""
    # Truncate inputs to keep prompt concise
    sources_summary = sources[:600] + "..." if len(sources) > 600 else sources
    answer_summary = answer[:300] + "..." if len(answer) > 300 else answer
    query_summary = query[:100] + "..." if len(query) > 100 else query
    
    # Simplified prompt: Focus on F1 score only
    judge_prompt = f"""RAG Evaluation Task:
Q: {query_summary}
Sources: {sources_summary}
Answer: {answer_summary}

Score:
F1: Harmonic mean of precision and recall (0.00 to 1.00)

Examples:
Q: Super Bowl score? Sources: Chiefs 38-35 Eagles. Answer: Chiefs won 38-35.
{{"f1":1.00}}

Q: Who won? Sources: Chiefs won. Answer: Eagles won.
{{"f1":0.00}}

Output JSON only: {{"f1":0.XX}} with 2 decimals."""

    try:
        # Token check
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        prompt_tokens = len(tokenizer.encode(judge_prompt))
        print(f"Judge Prompt Token Length: {prompt_tokens}")
        if prompt_tokens > 450:
            print("Warning: Prompt too long for T5-base. Consider flan-t5-large.")

        # Generate with beam search for coherence
        llm_output = generator(
            judge_prompt,
            max_new_tokens=30,  # Short output for JSON
            num_beams=4,        # Beam search for better coherence
            do_sample=False,    # Greedy with beams
            repetition_penalty=1.2,  # Avoid prompt repetition
            early_stopping=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )[0]["generated_text"].strip()
        
        # Clean output: Remove prompt echoes or invalid prefixes
        if any(phrase in llm_output.lower() for phrase in ["f1:", "score:", "json:"]):
            llm_output = re.sub(r'(?i)(f1|score|json):', '', llm_output).strip()
        
        # Ensure valid JSON format
        clean_output = "{" + llm_output.strip("{}") + "}" if not llm_output.startswith('{') else llm_output
        print(f"Judge Raw Output: {repr(clean_output)}")
        
        # Parse JSON
        judge_data = json.loads(clean_output)
        f1 = float(judge_data.get("f1", 0.0))
        
        return {
            "f1": round(f1, 2),
            "parsing_failed": False,
            "raw": llm_output,
            "prompt": judge_prompt[:250] + "..." if len(judge_prompt) > 250 else judge_prompt
        }
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Judge Parse Error: {e}")
        
        # Fallback: Extract F1 score via regex
        f1_match = re.search(r'(?:f1|f)\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)', llm_output, re.IGNORECASE)
        f1_val = float(f1_match.group(1)) if f1_match else None
        
        # Extract any float as a last resort
        floats = re.findall(r'([0-9]{1,3}(?:\.[0-9]{1,2})?)', llm_output)
        if floats:
            f1_val = float(floats[0])
        
        # Default F1 score if no valid extraction
        f1 = f1_val if f1_val is not None else 0.5
        
        return {
            "f1": round(f1, 2),
            "parsing_failed": True,
            "raw": llm_output,
            "prompt": judge_prompt[:250] + "...",
            "error": str(e)
        }

def load_chroma(embedder, persist_dir="./chroma_db"):
    """
    Load an existing Chroma DB from disk.
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"No Chroma DB found at {persist_dir}")
    
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)

# ---------- Main RAG Query Function ----------
def run_query(query, config):
    """
    Run the RAG query pipeline.
    """
    # Extract config
    k = config.get("k", 3)
    chunk_size = config.get("chunk_size", 1000)
    chunk_overlap = config.get("chunk_overlap", 200)
    rag_model = config.get("rag_model", "google/flan-t5-base")
    embed_model = config.get("embed_model", "BAAI/bge-base-en-v1.5")
    use_memory = config.get("use_memory", True)
    memory_obj = config.get("memory_obj")
    mcp_enabled = config.get("mcp_enabled", False)
    mcp_url = config.get("mcp_url")
    auto_refine = config.get("auto_refine", False)
    auto_extract = config.get("auto_extract", False)

    # MCP: Auto-refine or extract if enabled
    refined_query = query
    if mcp_enabled and mcp_url:
        try:
            if auto_refine:
                refine_resp = mcp_call_tool_sync(mcp_url, "refine_query", {"query": query})
                refined_query = refine_resp.get("refined_query", query)
                print(f"Refined query: {refined_query}")
            
            if auto_extract:
                extract_resp = mcp_call_tool_sync(mcp_url, "extract_entities", {"query": refined_query})
                print(f"Extracted entities: {extract_resp}")
        except Exception as e:
            print(f"MCP call failed: {e}")

    # Get DB
    db, embedder = get_db(embed_model_name=embed_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Retrieve
    retrieved_docs = db.similarity_search(refined_query, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Load generator
    generator = pipeline(
        "text2text-generation",
        model=rag_model,
        device=0 if torch.cuda.is_available() else -1
    )

    # Memory load
    chat_history = ""
    if use_memory and memory_obj:
        try:
            memory_vars = memory_obj.load_memory_variables({})
            chat_history = memory_vars.get("history", "")
            if chat_history:
                chat_history = f"History: {chat_history}\n"
        except Exception as e:
            print(f"Memory load error: {e}")
    
    prompt = f"{chat_history}Q: {refined_query}\nContext: {context}\nAnswer:"
    
    # Generate
    with torch.no_grad():
        answer_output = generator(
            prompt,
            max_new_tokens=512,  # Default max tokens for flan-t5
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id
        )[0]["generated_text"]
    
    answer = answer_output.split("Answer:")[-1].strip() if "Answer:" in answer_output else answer_output.strip()

    # Memory save
    if use_memory and memory_obj:
        try:
            memory_obj.save_context({"query": refined_query}, {"output": answer})
        except Exception as e:
            print(f"Memory save error: {e}")

    # Judge
    llm_judge = simple_llm_judge(generator, refined_query, context, answer)

    # Metrics
    try:
        detailed_metrics = evaluate_response_detailed(answer, context)
    except:
        detailed_metrics = {}

    # Sources
    sources = [doc.metadata.get("source", f"Doc {i+1}") for i, doc in enumerate(retrieved_docs)]

    return {
        "answer": answer,
        "sources": sources,
        "llm_judge": llm_judge,
        "detailed_metrics": detailed_metrics,
        "refined_query": refined_query if refined_query != query else None,
        "context_length": len(context)
    }