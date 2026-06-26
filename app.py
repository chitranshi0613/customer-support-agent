import streamlit as st
from langchain_core.messages import HumanMessage
from agent import agent, AgentState

st.title("TaskFlow Customer Support")
st.caption("AI-powered support agent with memory")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# User input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.write(prompt)

    # Run agent
    state = AgentState(messages=st.session_state.messages)
    result = agent.invoke(state)
    
    # Get latest AI response
    new_messages = result["messages"][len(st.session_state.messages):]
    
    for msg in new_messages:
        st.session_state.messages.append(msg)
        with st.chat_message("assistant"):
            st.write(msg.content)
            
    # Show escalation warning
    if result.get("escalate"):
        st.warning("⚠️ This conversation has been escalated to a human agent.")
        