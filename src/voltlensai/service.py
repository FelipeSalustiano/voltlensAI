import bentoml
from PIL import Image
from voltlensai.inference import load_model, predict


@bentoml.service
class MeterDetectionService:

    def __init__(self):
        self.model = load_model()

    @bentoml.api
    def predict(self, image: Image.Image, /) -> dict:
        result = predict(image, self.model)

        if result is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
            }

        return {
            "detected": True,
            "confidence": result["confidence"],
            "bounding_box": {
                "center_x": result["center_x"],
                "center_y": result["center_y"],
                "width": result["width"],
                "height": result["height"],
            },
        }