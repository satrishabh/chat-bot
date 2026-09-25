import streamlit as st
from snowflake.snowpark.context import get_active_session
import json

st.set_page_config(page_title="Cortex Chatbot", page_icon="🤖", layout="wide")

session = get_active_session()

MODELS = [
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    "mistral-large2",
    "snowflake-llama-3.3-70b",
]

with st.sidebar:
    st.title("Cortex Chatbot")
    selected_model = st.selectbox("Model", MODELS, index=0)
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful AI assistant. Be concise and clear in your responses.",
        height=120,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def call_cortex(model, conversation, temp):
    prompt_json = json.dumps(conversation)
    options_json = json.dumps({"temperature": temp})
    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, PARSE_JSON(?), PARSE_JSON(?)) AS resp"
    result = session.sql(sql, params=[model, prompt_json, options_json]).collect()
    raw = result[0]["RESP"]
    parsed = json.loads(raw)
    choice = parsed.get("choices", [{}])[0]
    msg_obj = choice.get("messages") or choice.get("message", "")
    if isinstance(msg_obj, dict):
        return msg_obj.get("content", str(msg_obj))
    return str(msg_obj) if msg_obj else raw


# Display conversation history
for msg in st.session_state.messages:
    role_label = "**You:**" if msg["role"] == "user" else "**Assistant:**"
    st.markdown(f"{role_label} {msg['content']}")
    st.markdown("---")

# Input area
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", height=100, key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})

    conversation = [{"role": "system", "content": system_prompt}]
    for m in st.session_state.messages:
        conversation.append({"role": m["role"], "content": m["content"]})

    with st.spinner("Thinking..."):
        try:
            answer = call_cortex(selected_model, conversation, temperature)
        except Exception as e:
            answer = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.experimental_rerun()
