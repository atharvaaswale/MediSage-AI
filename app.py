import streamlit as st
from src.engine import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

st.title("⚕️ MedSage AI")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag_chain = get_rag_chain()

# Display chat history for a continuous look
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a medical question:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # We pass the 'chat_history' stored in session_state to the chain
        response = rag_chain.invoke({
            "input": prompt,
            "chat_history": st.session_state.chat_history
        })
        
        answer = response["answer"]
        st.markdown(answer)
    
    # Update history with the latest turn
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer),
    ])
