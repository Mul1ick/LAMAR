from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# Importing your existing pipeline function
from doclamar.pipeline.run_pipeline import run_pipeline 

app = FastAPI(title="DocLamar Agent API")

# Allow requests from your Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Electron app's origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected data from React
class QueryRequest(BaseModel):
    query: str
    directory: str

@app.post("/chat")
async def process_chat(request: QueryRequest):
    print(f"\n🚀 [1/3] Received new query: '{request.query}'")
    print(f"📂 [2/3] Scanning directory: {request.directory}")
    print("🧠 [3/3] AI is thinking... (Please wait 30-60 seconds)")
    try:
        # Pass the React inputs directly into your pipeline [cite: 348]
        answer = run_pipeline(
            query=request.query, 
            root_path=request.directory, 
            top_k=5
        )
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)