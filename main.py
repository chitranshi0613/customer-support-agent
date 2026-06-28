from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import agent, AgentState

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Customer Support Agent is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    state = AgentState(messages=[HumanMessage(content=request.message)])
    result = agent.invoke(state)
    last_message = result["messages"][-1]
    return {"response": last_message.content}
