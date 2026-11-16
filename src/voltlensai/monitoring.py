import json
import logging
from pathlib import Path
from datetime import datetime, timezone


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "inference.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("voltlens.monitoring")
logger.setLevel(logging.INFO)
logger.propagate = False


if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a",
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)


def log_inference(
    request_id: str,
    endpoint: str,
    input_data: dict,
    output_data: dict | None,
    latency_ms: float,
    status_code: int,
    error: str | None = None,
):
    """
    Registra uma requisição de inferência em formato JSON.
    Cada chamada gera uma nova linha no arquivo.
    """

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        ),
        "request_id": request_id,
        "endpoint": endpoint,
        "input": input_data,
        "output": output_data,
        "latency_ms": round(latency_ms, 4),
        "status_code": status_code,
        "error": error,
    }

    logger.info(
        json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
