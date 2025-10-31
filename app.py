# app.py
import streamlit as st
import traceback
from mcp_client import mcp_list_tools_sync, mcp_call_tool_sync
from main import run_query
from utils.memory_utils import get_session_memory

st.set_page_config(page_title="RAG Sports QA App", layout="centered")
st.title("RAG Sports Event QA App")

# === SIDEBAR: SETTINGS ===
with st.sidebar:
    st.header("Settings")
    k_results = st.number_input("Retriever k (top docs)", min_value=1, max_value=10, value=3)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=5000, value=1000)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200)
    model_name = st.text_input("RAG model name (HuggingFace)", value="google/flan-t5-base")
    embed_model_name = st.text_input("Embedding model name", value="BAAI/bge-base-en-v1.5")
    use_memory = st.checkbox("Enable memory", value=True)

    mcp_enabled = st.checkbox("Enable MCP", value=False)
    mcp_url = None
    auto_refine = False
    auto_extract = False

    if mcp_enabled:
        mcp_url = st.text_input("MCP URL", value="http://127.0.0.1:8050/")
        auto_refine = st.checkbox("Auto-refine query", value=True)
        auto_extract = st.checkbox("Auto-extract entities", value=True)

        # Initialize persistent connection state (defaults to disconnected)
        if not hasattr(st.session_state, 'mcp_connected'):
            st.session_state.mcp_connected = False
        if not hasattr(st.session_state, 'mcp_tools'):
            st.session_state.mcp_tools = []

        # Always-visible status indicator
        status_emoji = "🟢" if st.session_state.mcp_connected else "🔴"
        status_text = "Connected" if st.session_state.mcp_connected else "Disconnected"
        status_container = st.container()
        with status_container:
            st.metric("MCP Status", f"{status_text}", help="🟢 Green = Connected | 🔴 Red = Disconnected")
            if st.session_state.mcp_tools:
                st.caption(f"Tools: {', '.join(st.session_state.mcp_tools)}")

        # Connect button (updates status)
        col_btn, col_refresh = st.columns([3, 1])
        with col_btn:
            if st.button("Connect to MCP"):
                try:
                    tools = mcp_list_tools_sync(mcp_url)
                    st.session_state.mcp_connected = True
                    st.session_state.mcp_tools = [t['name'] for t in tools]
                    st.rerun()  # Refresh to update the metric instantly
                except Exception as e:
                    st.session_state.mcp_connected = False
                    st.session_state.mcp_tools = []
                    st.error(f"🔴 Connection failed: {e}")
                    st.expander("Debug").write(traceback.format_exc())
                    st.rerun()

        with col_refresh:
            if st.button("↻", key="refresh_mcp", help="Re-check connection"):
                # Force a re-connection attempt
                try:
                    tools = mcp_list_tools_sync(mcp_url)
                    st.session_state.mcp_connected = True
                    st.session_state.mcp_tools = [t['name'] for t in tools]
                except Exception:
                    st.session_state.mcp_connected = False
                    st.session_state.mcp_tools = []
                st.rerun()

# === MAIN QUERY INTERFACE ===
query = st.text_input("Enter your question:", key="query_input")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            config = {
                "k": k_results,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "rag_model": model_name,
                "embed_model": embed_model_name,
                "use_memory": use_memory,
                "memory_obj": get_session_memory(),
                "mcp_enabled": mcp_enabled,
                "mcp_url": mcp_url,
                "auto_refine": auto_refine,
                "auto_extract": auto_extract
            }
            result = run_query(query, config)

        # === DISPLAY ANSWER ===
        st.markdown("### Answer:")
        st.info(result["answer"])

        st.markdown("**Sources:**")
        for i, src in enumerate(result["sources"], 1):
            st.write(f"{i}. {src}")

        # === LLM JUDGE ===
        if result.get("llm_judge"):
            judge = result["llm_judge"]
            if judge.get("parsing_failed"):
                st.warning("LLM Judge output was malformed.")
                with st.expander("Raw Judge Output"):
                    st.write(f"**Raw:** {judge.get('raw', '')}")
                    if 'prompt' in judge:
                        st.text_area("Prompt Preview", judge['prompt'][:400] + "...", height=100)
                    if 'error' in judge:
                        st.write(f"**Parse Error:** {judge['error']}")
                    st.write(f"**Full Keys:** {list(judge.keys())}")
            else:
                st.markdown("### LLM Judge")
                st.write(f"- F1 Score: {judge.get('f1', 'N/A')}")

        # === STORE RESULT FOR FEEDBACK ===
        st.session_state.last_result = result
        st.session_state.last_query = query

# === FEEDBACK SECTION (updated) ===
if mcp_enabled and mcp_url and "last_result" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 Feedback")
    st.write("Was this response helpful?")

    col1, col2 = st.columns([1, 1])

    # Thumbs Up (single-click, immediate feedback send)
    with col1:
        if st.button("👍 Yes", key="thumbs_up", help="Positive feedback"):
            try:
                resp = mcp_call_tool_sync(mcp_url, "send_feedback_email", {
                    "query": st.session_state.last_query,
                    "answer": st.session_state.last_result["answer"],
                    "feedback": "positive",
                    "comments": ""
                })
                st.success("Thanks — feedback sent.")
                # store minimal state for UI if needed
                st.session_state.last_feedback = {"type": "positive", "server_resp": resp}
            except Exception as e:
                st.error(f"Failed to send feedback: {e}")

    # Thumbs Down (toggles a compact, deterministic form)
    with col2:
        # initialize flag once
        if "show_neg_feedback" not in st.session_state:
            st.session_state.show_neg_feedback = False

        # toggle button to show/hide negative feedback form
        if not st.session_state.show_neg_feedback:
            if st.button("👎 No", key="thumbs_down", help="Negative feedback"):
                st.session_state.show_neg_feedback = True
        else:
            # compact bordered form using safe HTML (Streamlit doesn't support container(border=True))
            st.markdown(
                """
                <div style="
                    border:1px solid #e0e0e0;
                    padding:12px;
                    border-radius:8px;
                    background-color: #fafafa;
                ">
                <strong>Tell us what went wrong (optional)</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            comments = st.text_area(
                "", key="neg_feedback", height=90,
                placeholder="What was missing or incorrect?"
            )

            row_a, row_b, _ = st.columns([1, 1, 2])
            with row_a:
                if st.button("Submit", key="submit_neg", help="Send negative feedback"):
                    try:
                        resp = mcp_call_tool_sync(mcp_url, "send_feedback_email", {
                            "query": st.session_state.last_query,
                            "answer": st.session_state.last_result["answer"],
                            "feedback": "negative",
                            "comments": comments.strip() if comments else ""
                        })
                        st.success("Thanks — negative feedback sent.")
                        # hide form after success
                        st.session_state.show_neg_feedback = False
                        st.session_state.last_feedback = {"type": "negative", "comments": comments.strip(), "server_resp": resp}
                    except Exception as e:
                        st.error(f"Failed to send feedback: {e}")
            with row_b:
                if st.button("Cancel", key="cancel_neg", help="Close without sending"):
                    # simply hide the form; no rerun needed
                    st.session_state.show_neg_feedback = False
else:
    if "last_result" in st.session_state:
        st.info("Enable MCP to send feedback.")
