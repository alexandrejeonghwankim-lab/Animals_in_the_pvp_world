import json
from urllib.parse import urlencode
from urllib.request import urlopen

import config


INSPECT_APPID = 570


def main():


    with open('test_50.json', 'r') as file:
        data = json.load(file)


    res = data["response"]["apps"]
    for app in res:
        appid = app["appid"]

        params = {
            "appids": appid,
            "format": "json",
        }

        url = config.APP_DETAILS_URL + "?" + urlencode(params)

        with urlopen(url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        with open("details.json", 'a') as outfile:
            json.dump(data, outfile, indent=2)



if __name__ == "__main__":
    main()