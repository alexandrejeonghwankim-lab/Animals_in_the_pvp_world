import argparse
from pathlib import Path

import requests
import json

from config import url, output_file1, output_file2


def fetch_page(url: str) :
    """Return the page HTML and its parsed BeautifulSoup document."""
    try:
        response = requests.get(
            url,
            timeout=20,
        )
        response.raise_for_status()
        print("here", response)
    except requests.RequestException as e:
        print("Something went wrong:", e)
        return None
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and parse a web page.")
    parser.add_argument("-u", "--url", help="URL to scrape", default=url)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional file path for saving the downloaded HTML",
        default=output_file1
    )
    args = parser.parse_args()
    print("main, url : ", url)
    res = fetch_page(args.url)
    if not res:
        return
    data = res.json()

    for out_f in (output_file1, output_file2):
        with open(out_f, 'w') as file:
            json.dump(data, file, indent=2)
        
    
if __name__ == "__main__":
    main()
    