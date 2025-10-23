from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import TYPE_CHECKING

import numpy as np
import redis
import tritonclient.http.aio as triton
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from youarebot.data.schemas import IncomingMessage, Prediction

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
else:  # pragma: no cover - typing compatibility
    from typing import Awaitable, Callable  # noqa: UP035

logger = logging.getLogger(__name__)

security = HTTPBasic()
VALID_USERNAME = os.environ.get("FASTAPI_ADMIN_USERNAME", "admin")
VALID_PASSWORD = os.environ.get("FASTAPI_ADMIN_PASSWORD", "hardpass")

TRITON_HOST = os.environ.get("TRITON_HOST", "inference-service")
TRITON_HTTP_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))
TRITON_MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "bot_classifier")
TRITON_TIMEOUT = float(os.environ.get("TRITON_TIMEOUT", "5.0"))
TRITON_URL = f"{TRITON_HOST}:{TRITON_HTTP_PORT}"

TRITON_TEXT_INPUT = "text"
TRITON_PROBABILITY_OUTPUT = "probability"

TRITON_CLIENT = triton.InferenceServerClient(url=TRITON_URL)

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

app = FastAPI(title="Inference Service", description="Классификация диалогов")


def check_auth(credentials: HTTPBasicCredentials) -> None:
    """Validate provided HTTP Basic credentials."""
    if credentials.username != VALID_USERNAME or credentials.password != VALID_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def protect_docs(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Guard the documentation endpoints with HTTP Basic auth."""
    protected_paths = ["/docs", "/redoc", "/openapi.json"]

    if request.url.path in protected_paths:
        try:
            credentials = await security(request)
        except HTTPException:
            return Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)

        try:
            check_auth(credentials)
        except HTTPException:
            return Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)

    return await call_next(request)


def make_cache_key(text: str) -> str:
    """Return a cache key generated via SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.post("/predict")
async def predict(msg: IncomingMessage) -> Prediction:
    """Возвращает вероятность того, что в диалоге участвует бот."""
    cache_key = make_cache_key(msg.text)
    cached_prob = redis_client.get(cache_key)
    if cached_prob is not None:
        logger.info("Cache hit for key: %s", cache_key)
        return Prediction(
            id=uuid.uuid4(),
            message_id=msg.id,
            dialog_id=msg.dialog_id,
            participant_index=msg.participant_index,
            is_bot_probability=float(cached_prob),
        )

    infer_input = triton.InferInput(TRITON_TEXT_INPUT, [1, 1], "BYTES")
    infer_input.set_data_from_numpy(np.array([[msg.text.encode("utf-8")]]))
    requested_output = triton.InferRequestedOutput(TRITON_PROBABILITY_OUTPUT)
    result = await TRITON_CLIENT.infer(
        TRITON_MODEL_NAME,
        inputs=[infer_input],
        outputs=[requested_output],
        request_id=str(uuid.uuid4()),
        timeout=TRITON_TIMEOUT,
    )

    is_bot_probability = result.as_numpy(TRITON_PROBABILITY_OUTPUT).flatten().tolist()[0]

    redis_client.set(cache_key, is_bot_probability)

    prediction_id = uuid.uuid4()
    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability,
    )
