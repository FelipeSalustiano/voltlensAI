
import time
import uuid

import bentoml
from PIL import Image, ImageStat

from voltlensai.inference import load_model, predict
from voltlensai.monitoring import log_inference
from voltlensai.middleware import InferenceErrorLoggingMiddleware


@bentoml.service
class MeterDetectionService:

    def __init__(self):
        self.model = load_model()

    @bentoml.api
    def predict(self, image: Image.Image, /) -> dict:

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        original_width, original_height = image.size

        grayscale = image.convert("L")
        pixel_stats = ImageStat.Stat(grayscale)

        input_data = {
            "width": original_width,
            "height": original_height,
            "format": image.format,
            "pixel_mean": pixel_stats.mean[0],
            "pixel_std": pixel_stats.stddev[0],
        }

        result = predict(image, self.model)

        if result is None:
            response = {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
            }

        else:
            response = {
                "detected": True,
                "confidence": result["confidence"],
                "bounding_box": {
                    "center_x": result["center_x"],
                    "center_y": result["center_y"],
                    "width": result["width"],
                    "height": result["height"],
                },
            }

        latency_ms = (time.perf_counter() - start_time) * 1000

        log_inference(
            request_id=request_id,
            endpoint="/predict",
            input_data=input_data,
            output_data=response,
            latency_ms=latency_ms,
            status_code=200,
        )

        return response


MeterDetectionService.add_asgi_middleware(
    InferenceErrorLoggingMiddleware
)
