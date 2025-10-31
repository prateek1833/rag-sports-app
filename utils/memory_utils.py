# utils/memory_utils.py
import streamlit as st
from langchain_community.memory import ConversationBufferMemory  # <-- New correct import

def get_session_memory(key: str = "conversation_memory"):
    if "conversation_memory" not in st.session_state:
        try:
            mem = ConversationBufferMemory(memory_key="chat_history", input_key="query",
                                          output_key="answer", return_messages=False)
        except TypeError:
            mem = ConversationBufferMemory(return_messages=False)
        st.session_state[key] = mem
    return st.session_state[key]

def clear_session_memory(key: str = "conversation_memory"):
    if key in st.session_state:
        try: del st.session_state[key]
        except: pass
