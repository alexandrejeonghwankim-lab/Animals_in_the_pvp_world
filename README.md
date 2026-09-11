# Animals in the PvP World

2 cats, one penguin and one capybara

## Overview

This project is a Python-based data collection and analysis system for Steam API data. It fetches information about Steam applications and stores it for further analysis and processing.

## Project Structure

```
Animals_in_the_pvp_world/
├── README.md              # Project documentation
├── config.py              # Configuration settings
├── fetch_steamapi.py      # Steam API data fetching script
├── dags/                  # Apache Airflow DAGs (directed acyclic graphs)
└── data/                  # Data storage directory
```

## Key Files

### `config.py`
Configuration file containing:
- **API URLs**: Steam API endpoints for fetching app lists and details
  - `API_URL`: Main Steam API endpoint for app list
  - `APP_DETAILS_URL`: Steam store API endpoint for app details
- **Database**: SQLite database path (`data/steam_apps.db`)
- **Request Settings**:
  - `MAX_RESULTS`: Maximum results per request (50,000)
  - `REQUEST_DELAY_SECONDS`: Delay between requests (1 second)
- **Progress Tracking**:
  - `last_appid_fetched`: Last fetched app ID for resuming operations
  - `APPIDS_FILE`: JSON file storing app IDs (e.g., `upto50k.json`)
- **API Authentication**: `STEAM_API_KEY_NAME` for environment variable lookup

### `fetch_steamapi.py`
Main script for fetching Steam API data:

**Functions:**
- `load_api_key()`: Loads Steam API key from `.env` file
- `fetch_app_page(api_key)`: Fetches Steam app data with pagination support
- `main()`: Entry point that orchestrates the data fetching process

**Features:**
- Reads API key from `.env` file (format: `STEAM_API_KEY=your_key_here`)
- Fetches paginated Steam app data (games, DLC, software, videos, hardware)
- Saves results to JSON file for processing
- Handles pagination by tracking `last_appid` for subsequent requests
- Provides feedback on available results and pagination status

**Output:**
- Saves fetched data to `from250kup.json`

## Setup Instructions

### Prerequisites
- Python 3.x
- Steam API key (obtainable from [Steamworks](https://steamcommunity.com/dev))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alexandrejeonghwankim-lab/Animals_in_the_pvp_world.git
   cd Animals_in_the_pvp_world
   ```

2. Create a `.env` file in the root directory:
   ```
   STEAM_API_KEY=your_steam_api_key_here
   ```

3. Install dependencies (if needed):
   ```bash
   pip install -r requirements.txt  # if available
   ```

### Running the Script

```bash
python fetch_steamapi.py
```

This will:
1. Load your Steam API key from the `.env` file
2. Fetch Steam app data (limited to 50,000 results per request)
3. Save the results to `from250kup.json`
4. Display pagination information for fetching remaining data

## Important Notes

- The Steam API has a **50,000 result limit per request**, requiring multiple requests with pagination
- Each app returns a `last_modified` timestamp to track changes
- A `price_change_number` field indicates if prices have changed since the last fetch
- Requests are rate-limited with a 1-second delay to respect API quotas

## Technology Stack

- **Language**: Python 100%
- **Data Processing**: Apache Airflow (DAGs directory for workflow orchestration)
- **Database**: SQLite
- **API**: Steam Web API

## Development

The `dags/` directory is set up for Apache Airflow workflows to automate and schedule data fetching and processing tasks.

The `data/` directory stores all project data and databases.

## License

[License information to be added]

## Authors

- Created by alexandrejeonghwankim-lab

---

For more information about the Steam API, visit the [Steam Web API documentation](https://developer.valvesoftware.com/wiki/Steam_Web_API).
