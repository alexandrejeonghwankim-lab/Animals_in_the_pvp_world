# Animals in the PvP World

2 cats, one penguin and one capybara

## Overview

This project is a Python-based data collection and analysis system for Steam API data. It fetches information about Steam applications and stores it for further analysis and processing. Automation implemented using Airflow.

### First Step

Getting app ids from Steam API. (50k per request)

### Second Step

Creating a database for raw data.

### Third Step

Getting apps details and storing them in the db. (200 per request, once every 6 minutes)  Scheduled with Airflow.

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
- `load_api_key()`: Loads Steam API key from a **local** `.env` file
- `fetch_app_page(api_key)`: Fetches Steam app data with pagination support
- `main()`: Entry point that orchestrates the data fetching process

**Features:**
- Reads API key from `.env` file (format: `STEAM_API_KEY=your_key_here`)
- Fetches paginated Steam app data (games, DLC, software, videos, hardware)
- Saves results to JSON file for processing
- Handles pagination by tracking `last_appid` for subsequent requests
- Provides feedback on available results and pagination status

**Output:**
- Saves fetched data to a new json file. Results in multiple files containing up to 50k objects.

### `create_db.py`
Main DAG for creating the raw details db:

**Task:**
- `create_database_tables()`: Create the records tables.

**Features:**
- Connects to the db path with SQLite3
- Creates the **records** table *if it does not exists*
- With 3 columns : id, data (string of json), date_collected (auto-fill)

**Output:**
- *Database and table created successfully*
### `fetch_details.py`
Main DAG fortgetting the raw details of each steam app on a schedule of 200 every 6 minutes :

**Task:**
- `fetch_steam_batch()`: Gets the next 200 apps' details.

**Features:**
- Reads a json file of appids and identify the next 200 to fetch thanks to an Airflow Variable storing the last used
- Becomes a skipped DAG if there is no more appids details to fetch (and stops by accessing airflow metadata and modifying it but bad practice to be removed)
- Calls app details, one at a time, and stores it in the raw database.
- Stops if code *429* returned
- Stores failed appids requests in a txt file

**Output:**
- *Batch completed. Successfully processed {successful_calls} apps.*
- *Updated Airflow Variable 'steam_last_appid_fetched' to {current_last_id}.*
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

3. Install dependencies:
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
3. Save the results to `afile.json`
4. Display pagination information for fetching remaining data


### Running Airflow in development
 
```bash
airflow standalone
```

Authentification: find auto-generated *username* and *password* in `simple_auth_manager_passwords.json.generated`

You can then make use of the Airflow UI to trigger and managed DAGS.

## Important Notes

- The Steam API has a **50,000 result limit per request**, requiring multiple requests with pagination
- Each app returns a `last_modified` timestamp to track changes
- A `price_change_number` field indicates if prices have changed since the last fetch
- Requests are rate-limited with a 1-second delay to respect API quotas

- The requests for details are limited to **200 every 5 minutes**
## Technology Stack

- **Language**: Python 100%
- **Data Processing**: Apache Airflow (DAGs directory for workflow orchestration)
- **Database**: SQLite
- **API**: Steam Web API

## Development

The `dags/` directory is set up for Apache Airflow workflows to automate and schedule data fetching and processing tasks.

The `data/` directory stores all project data and databases.



## Authors

-  [Alex Kim](https://github.com/alexandrejeonghwankim-lab)

-  [Max H.](https://github.com/Max96H)
- [Victor Courtois](https://github.com/VictorCourtois135)

- [Sooyoung Lee](https://github.com/patoobyte) 
---

For more information about the Steam API, visit the [Steam Web API documentation](https://developer.valvesoftware.com/wiki/Steam_Web_API).
