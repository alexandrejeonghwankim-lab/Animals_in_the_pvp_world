
from pathlib import Path


# API_URL = "https://partner.steam-api.com/IStoreService/GetAppList/v1/"
API_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
DATABASE_PATH = Path("data") / "steam_apps.db"
MAX_RESULTS = 50000
REQUEST_DELAY_SECONDS = 1
STEAM_API_KEY_NAME = "STEAM_API_KEY"

last_appid_fetched = 18521
APPIDS_FILE = "upto50k.json"

"""
The number of results is limited, so multiple requests should be made to obtain the full list. The results are returned in order of appid, 
subsequent requests should pass the last appid returned by the previous request as the last_appid parameter.

Items return a last_modified unix timestamp, which indicates the last time some change was made to that game's information 
or price (not all of these changes may be visible on the store). if_modified_since can be used to filter to only games that have 
been changed or added since a specific timestamp.

Items also return a price_change_number field. If this number has changed since a previous call, it indicates the price on that
 item may have changed, such as a discount starting or price adjustment.
"""