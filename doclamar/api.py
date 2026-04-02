from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from graph.builder import build_graph # ✅ Correctly importing your LangGraph builder

app = FastAPI(title="DocLamar Agent API")

# Allow requests from your Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected data from React
class QueryRequest(BaseModel):
    query: str
    directory: str

# Initialize the AI graph once when the server starts
ai_graph = build_graph()

@app.post("/chat")
async def process_chat(request: QueryRequest):
    print(f"\n🚀 [1/3] Received new query: '{request.query}'")
    print(f"📂 [2/3] Scanning directory: {request.directory}")
    print("🧠 [3/3] AI is thinking... (Please wait 30-60 seconds)")
    
    try:
        # 1. Setup the initial state required by your LangGraph nodes
        initial_state = {
            "query": request.query,
            "root_path": request.directory,
            "top_k": 5,
            "routing_plan": None,
            "candidate_documents": None,
            "parsed_chunks": None,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "final_answer": None,
            "source_files": None,
            "citations": None,
            "search_stats": None,
            "evaluation": None,
            "error": None,
            "retry_count": 0,
            "node_timings": {},
        }

        # 2. Run the request through your AI pipeline
        result = ai_graph.invoke(initial_state)

        # 3. Extract the answer (fallback to an error message if it fails)
        answer = result.get("final_answer") or "I couldn't generate an answer for that."
        raw_citations = result.get("citations") or []

        formatted_citations = [
            {"file": c.get("file_name", "Unknown"),"path": c.get("file_path", ""), # <--- ADD THIS LINE,
            "snippet": c.get("preview", "")}
            for c in raw_citations
        ]

        # 4. Return exactly what your App.jsx expects (data.response)
        return {
            "response": answer,
            "citations": formatted_citations
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import uuid
from chat.document_chat import load_document, chat as doc_chat

# 1. Store active file sessions in memory
active_sessions = {}

class ChatLoadRequest(BaseModel):
    file_path: str

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str

# 2. Endpoint to load and index a single document
@app.post("/chat/load")
async def load_single_doc(request: ChatLoadRequest):
    try:
        print(f"📄 Loading specific document: {request.file_path}")
        session = load_document(request.file_path)
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = session
        return {"session_id": session_id, "file_name": session.file_name}
    except Exception as e:
        print(f"❌ Error loading doc: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 3. Endpoint to chat with the loaded document
@app.post("/chat/message")
async def doc_message(request: ChatMessageRequest):
    session = active_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found")
    
    print(f"💬 Chatting with {session.file_name}...")
    result = doc_chat(session, request.message)
    
    # Format citations to match your frontend
    formatted_citations = [
        {
            "file": session.file_name, 
            "path": session.file_path, 
            "snippet": c.get("preview", "")
        }
        for c in result.get("citations", [])
    ]
    
    return {"response": result["answer"], "citations": formatted_citations}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)