
import io
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageEnhance


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = BASE_DIR / "images"

API_URL = "http://localhost:3000/predict"

ORIGINAL_REQUESTS = 70
TRANSFORMED_REQUESTS = 20
INVALID_REQUESTS = 10

TIMEOUT = 60

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_images():
    images = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em {IMAGE_DIR}"
        )

    return sorted(images)


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def prepare_original(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image_to_bytes(image)


def prepare_transformed(image_path, index):
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        brightness = [0.7, 0.85, 1.15, 1.3][index % 4]
        contrast = [0.8, 0.95, 1.1, 1.25][index % 4]

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)

        return image_to_bytes(image)


def send_request(payload, valid=True):
    headers = {
        "Content-Type": "image/jpeg"
    }

    request = Request(
        API_URL,
        data=payload,
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            status_code = response.status
            response.read()

    except HTTPError as error:
        status_code = error.code
        error.read()

    except URLError as error:
        status_code = None
        print(f"Erro de conexão: {error.reason}")

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "status_code": status_code,
        "latency_ms": latency_ms,
        "valid_input": valid,
    }


def main():
    images = find_images()

    print(f"Imagens encontradas: {len(images)}")
    print(f"API: {API_URL}")
    print("Iniciando teste de carga...\n")

    results = []

    print("Enviando imagens originais...")

    for index in range(ORIGINAL_REQUESTS):
        image_path = images[index % len(images)]
        payload = prepare_original(image_path)

        result = send_request(payload)
        results.append(result)

        print(
            f"[{len(results):03d}/100] "
            f"Original | HTTP {result['status_code']} | "
            f"{result['latency_ms']:.2f} ms"
        )

    print("\nEnviando imagens transformadas...")

    for index in range(TRANSFORMED_REQUESTS):
        image_path = images[index % len(images)]
        payload = prepare_transformed(image_path, index)

        result = send_request(payload)
        results.append(result)

        print(
            f"[{len(results):03d}/100] "
            f"Transformada | HTTP {result['status_code']} | "
            f"{result['latency_ms']:.2f} ms"
        )

    print("\nEnviando requisições inválidas...")

    for index in range(INVALID_REQUESTS):
        payload = f"invalid image {index}".encode("utf-8")

        result = send_request(payload, valid=False)
        results.append(result)

        print(
            f"[{len(results):03d}/100] "
            f"Inválida | HTTP {result['status_code']} | "
            f"{result['latency_ms']:.2f} ms"
        )

    successful = [
        item for item in results
        if item["status_code"] is not None
        and 200 <= item["status_code"] < 400
    ]

    failed = [
        item for item in results
        if item["status_code"] is None
        or item["status_code"] >= 400
    ]

    successful_latencies = [
        item["latency_ms"] for item in successful
    ]

    total = len(results)

    print("\n" + "=" * 45)
    print("RESUMO DO TESTE DE CARGA")
    print("=" * 45)

    print(f"Total de requisições: {total}")
    print(f"Sucessos: {len(successful)}")
    print(f"Falhas: {len(failed)}")

    print(
        f"Taxa de erro: "
        f"{len(failed) / total * 100:.2f}%"
    )

    if successful_latencies:
        print(
            f"Latência média (cliente): "
            f"{statistics.mean(successful_latencies):.2f} ms"
        )
        print(
            f"Latência mediana (cliente): "
            f"{statistics.median(successful_latencies):.2f} ms"
        )
        print(
            f"Latência mínima (cliente): "
            f"{min(successful_latencies):.2f} ms"
        )
        print(
            f"Latência máxima (cliente): "
            f"{max(successful_latencies):.2f} ms"
        )

    print("=" * 45)


if __name__ == "__main__":
    main()
