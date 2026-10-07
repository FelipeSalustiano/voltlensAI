from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from voltlensai.model import MeterNetwork


MODEL_IMAGE_SIZE = (360, 480)
MODEL_PATH = Path("model/meter_detector.pt")


def load_model():
    model = MeterNetwork()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def decode_best_box(prediction, threshold=0.1):
    prediction = prediction[0]

    confidence = prediction[4].sigmoid()

    best_index = confidence.argmax()

    grid_y, grid_x = divmod(
        best_index.item(),
        confidence.shape[1],
    )

    score = confidence[grid_y, grid_x].item()

    if score < threshold:
        return None

    grid_h, grid_w = confidence.shape

    box = prediction[:4, grid_y, grid_x].sigmoid()

    local_x = box[0].item()
    local_y = box[1].item()
    width = box[2].item()
    height = box[3].item()

    center_x = (grid_x + local_x) / grid_w
    center_y = (grid_y + local_y) / grid_h

    return {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "confidence": score,
    }


def predict(image: Image.Image, model):
    image = image.convert("RGB")
    image = image.resize(MODEL_IMAGE_SIZE)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor)

    return decode_best_box(prediction)