
import time
import uuid

from voltlensai.monitoring import log_inference


class InferenceErrorLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["path"] != "/predict"
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        status_code = 500

        headers = dict(scope.get("headers", []))

        input_data = {
            "content_type": headers.get(b"content-type", b"").decode(
                "latin-1"
            ),
            "content_length": headers.get(b"content-length", b"").decode(
                "latin-1"
            ),
        }

        async def send_wrapper(message):
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000

            log_inference(
                request_id=request_id,
                endpoint="/predict",
                input_data=input_data,
                output_data=None,
                latency_ms=latency_ms,
                status_code=status_code,
                error=f"{type(exc).__name__}: {str(exc)}",
            )
            raise

        if status_code >= 400:
            latency_ms = (time.perf_counter() - start_time) * 1000

            log_inference(
                request_id=request_id,
                endpoint="/predict",
                input_data=input_data,
                output_data=None,
                latency_ms=latency_ms,
                status_code=status_code,
                error=f"HTTP {status_code} returned by BentoML",
            )
