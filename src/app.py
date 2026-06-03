from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from .rag_pipeline import MentalHealthRAG


class ChatRequest(BaseModel):
    query: str


class Resource(BaseModel):
    score: float
    page_content: str
    response: str


class ChatResponse(BaseModel):
    answer: str
    resources: List[Resource]


app = FastAPI(
    title="Mental Health RAG Chatbot",
    description="A FastAPI chatbot that uses MentalHealthRAG for retrieval-augmented mental health responses.",
    version="0.1.0",
)

rag: MentalHealthRAG | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global rag
    rag = MentalHealthRAG()
    documents = rag.load_and_preprocess()
    rag.setup_retriever(documents)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global rag
    if rag is not None:
        rag.close()


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized.")

    result = rag.query(request.query)
    if not result or "answer" not in result:
        raise HTTPException(status_code=500, detail="Failed to generate a response.")

    return ChatResponse(
        answer=result["answer"],
        resources=result.get("resources", []),
    )
