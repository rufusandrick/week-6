from __future__ import annotations

import asyncio
import hashlib
import sys
import types
import uuid
from collections.abc import AsyncGenerator


def _ensure_fastapi_stub() -> None:
    if "fastapi" in sys.modules:
        return

    fastapi_module = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str | None = None) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Response:
        def __init__(self, headers: dict[str, str] | None = None, status_code: int = 200) -> None:
            self.headers = headers or {}
            self.status_code = status_code

    class Request:
        def __init__(self, path: str) -> None:
            self.url = types.SimpleNamespace(path=path)

    class FastAPI:
        def __init__(self, *args, **kwargs) -> None:
            self._routes: dict[str, object] = {}

        def middleware(self, _name: str):
            def decorator(func):
                return func

            return decorator

        def post(self, path: str):
            def decorator(func):
                self._routes[path] = func
                return func

            return decorator

    security_module = types.ModuleType("fastapi.security")

    class HTTPBasicCredentials:
        def __init__(self, username: str, password: str) -> None:
            self.username = username
            self.password = password

    class HTTPBasic:
        async def __call__(self, request: Request) -> HTTPBasicCredentials:  # pragma: no cover - default guard
            raise HTTPException(status_code=401)

    security_module.HTTPBasic = HTTPBasic
    security_module.HTTPBasicCredentials = HTTPBasicCredentials

    fastapi_module.FastAPI = FastAPI
    fastapi_module.HTTPException = HTTPException
    fastapi_module.Request = Request
    fastapi_module.Response = Response
    fastapi_module.security = security_module

    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.security"] = security_module


def _ensure_pydantic_stub() -> None:
    if "pydantic" in sys.modules:
        return

    import uuid as _uuid

    pydantic_module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **data) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    pydantic_module.BaseModel = BaseModel
    pydantic_module.StrictStr = str
    pydantic_module.UUID4 = _uuid.UUID

    sys.modules["pydantic"] = pydantic_module


def _ensure_redis_stub() -> None:
    if "redis" in sys.modules:
        return

    redis_module = types.ModuleType("redis")

    class Redis:
        def __init__(self, *args, **kwargs) -> None:
            self.store: dict[str, str] = {}

        def get(self, key: str) -> str | None:
            return self.store.get(key)

        def set(self, key: str, value: str) -> None:
            self.store[key] = value

    redis_module.Redis = Redis
    sys.modules["redis"] = redis_module


_ensure_fastapi_stub()
_ensure_pydantic_stub()
_ensure_redis_stub()


try:  # pragma: no cover - only executed when numpy is available
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback stub

    class _Array(list):
        def flatten(self) -> "_Array":
            def _flatten(items):
                for item in items:
                    if isinstance(item, (list, tuple, _Array)):
                        yield from _flatten(item)
                    else:
                        yield item

            return _Array(_flatten(self))

        def tolist(self) -> list:
            def _convert(items):
                if isinstance(items, _Array):
                    return [_convert(value) for value in items]
                if isinstance(items, list):
                    return [_convert(value) for value in items]
                return items

            return _convert(self)

    def _array(data):  # type: ignore[override]
        return _Array(data)

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.array = _array
    numpy_stub.ndarray = _Array
    numpy_stub.isscalar = lambda obj: isinstance(obj, (int, float))
    numpy_stub.bool_ = bool
    sys.modules["numpy"] = numpy_stub
    np = numpy_stub

import pytest
from fastapi.security import HTTPBasicCredentials


def _ensure_triton_stub() -> None:
    """Provide a lightweight stub for the triton client if it is missing."""

    if "tritonclient.http.aio" in sys.modules:
        return

    triton_module = types.ModuleType("tritonclient")
    http_module = types.ModuleType("tritonclient.http")
    aio_module = types.ModuleType("tritonclient.http.aio")

    class _InferInput:
        def __init__(self, name: str, shape: list[int], dtype: str) -> None:
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.data = None

        def set_data_from_numpy(self, array: np.ndarray) -> None:
            self.data = array

    class _InferRequestedOutput:
        def __init__(self, name: str) -> None:
            self.name = name

    class _InferenceServerClient:
        def __init__(self, url: str) -> None:  # pragma: no cover - safety net
            self.url = url

        async def infer(self, *args, **kwargs):  # pragma: no cover - unused guard
            msg = "Stubbed client cannot perform inference"
            raise RuntimeError(msg)

    aio_module.InferInput = _InferInput
    aio_module.InferRequestedOutput = _InferRequestedOutput
    aio_module.InferenceServerClient = _InferenceServerClient

    triton_module.http = http_module
    http_module.aio = aio_module

    sys.modules["tritonclient"] = triton_module
    sys.modules["tritonclient.http"] = http_module
    sys.modules["tritonclient.http.aio"] = aio_module


_ensure_triton_stub()

from youarebot.api import main as main_module


class DummyRedis:
    """Simple in-memory Redis replacement."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(self, key: str, value: float) -> None:
        self.store[key] = str(value)


class DummyTritonResult:
    def __init__(self, value: float) -> None:
        self._value = value

    def as_numpy(self, name: str) -> np.ndarray:  # pragma: no cover - name unused in tests
        return np.array([[self._value]])


class DummyTritonClient:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[str, tuple, dict]] = []

    async def infer(self, model_name: str, *args, **kwargs) -> DummyTritonResult:
        self.calls.append((model_name, args, kwargs))
        return DummyTritonResult(self.value)


@pytest.fixture
def dummy_redis(monkeypatch: pytest.MonkeyPatch) -> DummyRedis:
    cache = DummyRedis()
    monkeypatch.setattr(main_module, "redis_client", cache)
    return cache


@pytest.fixture
def override_triton_client(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[DummyTritonClient, None]:
    client = DummyTritonClient(0.42)
    monkeypatch.setattr(main_module, "TRITON_CLIENT", client)
    yield client


@pytest.mark.parametrize(
    ("username", "password", "should_raise"),
    (
        (main_module.VALID_USERNAME, main_module.VALID_PASSWORD, False),
        ("invalid", main_module.VALID_PASSWORD, True),
        (main_module.VALID_USERNAME, "invalid", True),
    ),
)
def test_check_auth(username: str, password: str, should_raise: bool) -> None:
    credentials = HTTPBasicCredentials(username=username, password=password)

    if should_raise:
        with pytest.raises(main_module.HTTPException) as exc_info:
            main_module.check_auth(credentials)
        assert exc_info.value.status_code == 401
    else:
        main_module.check_auth(credentials)


def test_make_cache_key() -> None:
    text = "hello world"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert main_module.make_cache_key(text) == expected


def test_docs_require_auth() -> None:
    call_next_called = False

    async def call_next(request: main_module.Request) -> main_module.Response:  # pragma: no cover - should not run
        nonlocal call_next_called
        call_next_called = True
        return main_module.Response(status_code=200)

    request = main_module.Request("/docs")
    response = asyncio.run(main_module.protect_docs(request, call_next))

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Basic"
    assert not call_next_called


def test_docs_with_valid_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    class AllowSecurity:
        async def __call__(self, request: main_module.Request) -> main_module.HTTPBasicCredentials:
            return main_module.HTTPBasicCredentials(
                username=main_module.VALID_USERNAME,
                password=main_module.VALID_PASSWORD,
            )

    monkeypatch.setattr(main_module, "security", AllowSecurity())

    call_next_invoked = {"value": False}

    async def call_next(request: main_module.Request) -> main_module.Response:
        call_next_invoked["value"] = True
        return main_module.Response(status_code=200)

    response = asyncio.run(main_module.protect_docs(main_module.Request("/docs"), call_next))

    assert response.status_code == 200
    assert call_next_invoked["value"] is True


def test_predict_uses_cache(dummy_redis: DummyRedis, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_value = 0.75
    message = {
        "text": "cached text",
        "dialog_id": str(uuid.uuid4()),
        "id": str(uuid.uuid4()),
        "participant_index": 1,
    }

    cache_key = main_module.make_cache_key(message["text"])
    dummy_redis.set(cache_key, cache_value)

    class FailingClient:
        async def infer(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("Inference client should not be invoked when cache hits")

    monkeypatch.setattr(main_module, "TRITON_CLIENT", FailingClient())

    incoming = main_module.IncomingMessage(**message)
    prediction = asyncio.run(main_module.predict(incoming))

    assert prediction.is_bot_probability == pytest.approx(cache_value)
    assert str(prediction.message_id) == message["id"]


def test_predict_invokes_triton(dummy_redis: DummyRedis, override_triton_client: DummyTritonClient) -> None:
    message = {
        "text": "new text",
        "dialog_id": str(uuid.uuid4()),
        "id": str(uuid.uuid4()),
        "participant_index": 2,
    }

    incoming = main_module.IncomingMessage(**message)
    prediction = asyncio.run(main_module.predict(incoming))

    assert prediction.is_bot_probability == pytest.approx(override_triton_client.value)
    # Ensure result was cached for future requests
    cache_key = main_module.make_cache_key(message["text"])
    assert dummy_redis.get(cache_key) == str(override_triton_client.value)
    # Verify Triton client was invoked with the configured model name
    assert override_triton_client.calls
    assert override_triton_client.calls[0][0] == main_module.TRITON_MODEL_NAME
