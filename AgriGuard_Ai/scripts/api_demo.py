"""Call the AI service through a real HTTP client and print the exchange.

    python scripts/api_demo.py --endpoint irrigation
    python scripts/api_demo.py --endpoint crop-health

Starts uvicorn in a subprocess, waits for /v1/health, sends a real request and
prints the request and the raw response. Used to produce evidence/10_* and
evidence/11_*, and handy for checking the contract from a terminal.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agriguard.config import CROP_HEALTH_DIR, PROJECT_ROOT  # noqa: E402

IRRIGATION_PAYLOAD = {
    "timestamp": "2026-09-27T14:20:00Z",
    "device_id": "ESP32_01",
    "plot_id": "PLOT_A",
    "soil_moisture": 20.0,
    "temperature": 33.0,
    "humidity": 42.0,
    "light": 68000.0,
    "npk_n": 42.0,
    "npk_p": 27.0,
    "npk_k": 51.0,
    "water_flow": 0.0,
    "pump_status": 0,
    "crop_type": "tomato",
}


def wait_for_health(base_url, timeout=90):
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/v1/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:  # noqa: BLE001 - server not up yet
            pass
        time.sleep(1.0)
    raise SystemExit("AI service did not become healthy in time")


def pick_test_image():
    manifest = CROP_HEALTH_DIR / "splits.csv"
    if not manifest.exists():
        return None
    import csv

    with open(manifest, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
    if not rows:
        return None
    return CROP_HEALTH_DIR / sorted(rows, key=lambda r: r["key"])[0]["key"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=["irrigation", "crop-health", "both"], default="both")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    import httpx

    base_url = f"http://{args.host}:{args.port}"
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "agriguard.api:app",
            "--host", args.host, "--port", str(args.port), "--log-level", "warning",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health = wait_for_health(base_url)
        print(f"GET {base_url}/v1/health -> 200")
        print(json.dumps(health, indent=2))

        if args.endpoint in {"irrigation", "both"}:
            print(f"\nPOST {base_url}/v1/predict/irrigation")
            print("request body:")
            print(json.dumps(IRRIGATION_PAYLOAD, indent=2))
            response = httpx.post(
                f"{base_url}/v1/predict/irrigation", json=IRRIGATION_PAYLOAD, timeout=30
            )
            print(f"-> {response.status_code}")
            print(json.dumps(response.json(), indent=2))

        if args.endpoint in {"crop-health", "both"}:
            image = pick_test_image()
            if image is None:
                print("\nno crop-health test image available; run prepare_crop_dataset.py")
            else:
                print(f"\nPOST {base_url}/v1/predict/crop-health  (file: {image.name})")
                with open(image, "rb") as handle:
                    response = httpx.post(
                        f"{base_url}/v1/predict/crop-health",
                        files={"image": (image.name, handle.read(), "image/jpeg")},
                        timeout=120,
                    )
                print(f"-> {response.status_code}")
                try:
                    print(json.dumps(response.json(), indent=2))
                except Exception:  # noqa: BLE001
                    print(response.text)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover
            server.kill()


if __name__ == "__main__":
    main()
