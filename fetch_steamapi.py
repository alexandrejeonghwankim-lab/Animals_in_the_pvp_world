import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import config


def load_api_key():
    env_path = Path(".env")
    if not env_path.exists():
        return None

    for line in env_path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith(f"{config.STEAM_API_KEY_NAME}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def fetch_app_page(api_key):
    input_data = {
        "max_results": config.MAX_RESULTS,
        "include_games": True,
        "include_dlc": True,
        "include_software": True,
        "include_videos": True,
        "include_hardware": True,
    }

    params = {
        "key": api_key,
        "input_json": json.dumps(input_data),
        "format": "json",
    }

    url = config.API_URL + "?" + urlencode(params)

    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    api_key = load_api_key()

    if not api_key:
        print("Missing STEAM_API_KEY.")
        print("Create a .env file with this line:")
        print("STEAM_API_KEY=your_key_here")
        return

    data = fetch_app_page(api_key)
    print("printing...")
    with open("test_50.json", 'w') as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    main()