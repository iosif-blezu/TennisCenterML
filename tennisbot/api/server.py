from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from tennisbot.agent.router import get_router_agent
from tennisbot.config import get_settings
import os, sys


print("CWD =", os.getcwd())
print("sys.path[0] =", sys.path[0])
print("Modules visible:", os.listdir(sys.path[0]))


cfg = get_settings()

class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    output: str


app = FastAPI(title="TennisBot Simple Server")
agent = None  # will be set at startup
print("Open api key:", cfg.OPENAI_API_KEY)
@app.on_event("startup")
def load_agent():
    global agent
    agent = get_router_agent()

def _extract_output(result):
    return result.get("output") if isinstance(result, dict) else str(result)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="`input` must be non-empty")
    try:
        # off-load blocking invoke
        result = await run_in_threadpool(agent.invoke, {"input": request.input})
        output = _extract_output(result)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
    if not output:
        raise HTTPException(status_code=500, detail="Agent returned empty response")
    return ChatResponse(output=output)
