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

        # 4. Return exactly what your App.jsx expects (data.response)
        return {"response": answer}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)