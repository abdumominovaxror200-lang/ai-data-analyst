from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agent.agent import DataAnalystAgent
from app.agent.providers import LLMProviderError, build_provider_from_settings
from app.datasets.storage import DatasetNotFoundError, get_dataset_store
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    store = get_dataset_store()
    try:
        record = store.get(request.dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        provider = build_provider_from_settings(record.df)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"AI provider not configured: {exc}") from exc

    agent = DataAnalystAgent(provider)
    history = [{"role": m.role, "content": m.content} for m in request.history]
    try:
        result = agent.ask(record, request.message, history)
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info("chat dataset_id=%s tool_calls=%d", request.dataset_id, len(result["tool_calls"]))
    return ChatResponse(answer=result["answer"], tool_calls=result["tool_calls"], charts=result["charts"], data_caveats=result["data_caveats"], limitations=result["limitations"])
