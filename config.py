url = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=440&count=3"
output_file1 = "out.json"
output_file2 = "out.txt"


backup = "https://steamspy.com/api.php?request=top100in2weeks"

prefix_url = "api.steampowered.com"

from pathlib import Path


# API_URL = "https://partner.steam-api.com/IStoreService/GetAppList/v1/"
API_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APP_DETAILS_URL = "https://store.steampowered.com/api/appdetails"
DATABASE_PATH = Path("data") / "steam_apps.db"
MAX_RESULTS = 50
REQUEST_DELAY_SECONDS = 1
STEAM_API_KEY_NAME = "STEAM_API_KEY"