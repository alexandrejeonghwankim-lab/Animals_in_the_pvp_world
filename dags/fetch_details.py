import json
import sqlite3
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from pendulum import datetime

from airflow.sdk import DAG, Variable, task
from airflow.models import DagModel
from airflow.utils.session import create_session
from airflow.sdk.exceptions import AirflowSkipException

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Absolute paths based on project root
DB_PATH = PROJECT_ROOT / "data" / "steam_raw_data.db"
APPIDS_FILE_PATH = PROJECT_ROOT / "data" / getattr(config, "APPIDS_FILE", "apps.json")
FAILED_LOG_PATH = PROJECT_ROOT / "data" / "failed_detail_calls.txt"

PAUSE_BETWEEN_CALLS = 0.1  # Polite pause between single calls (seconds)
BATCH_SIZE = 200
FAIL_LIMIT = 5

with DAG(
    dag_id="01_fetch_steam_app_details",
    start_date=datetime(2026, 9, 1),
    schedule="*/6 * * * *",  # Runs every 6 minutes
    catchup=False,
    max_active_runs=1,  # Ensures runs never overlap
    tags=["steam", "ingestion"],
) as dag:

    @task
    def fetch_steam_batch():
        # Fetch persistent state (Airflow Variable -> config fallback -> 0)
        default_start_id = getattr(config, "last_appid_fetched", 10)
        last_fetched_id = int(Variable.get("steam_last_appid_fetched", default=default_start_id))

        print(f"Starting execution. Last App ID processed: {last_fetched_id}")

        # Load app IDs list
        if not APPIDS_FILE_PATH.exists():
            raise FileNotFoundError(f"App IDs file not found at {APPIDS_FILE_PATH}")

        with open(APPIDS_FILE_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        app_list = raw_data.get("response", {}).get("apps", [])

        # Locate slice of 200 items after last_fetched_id
        start_index = 0
        if last_fetched_id > 0:
            for idx, item in enumerate(app_list):
                if item.get("appid") == last_fetched_id:
                    start_index = idx + 1
                    break

        batch_to_process = app_list[start_index : start_index + BATCH_SIZE]

        if not batch_to_process:
            print("No new app IDs left to process. Pausing DAG permanently...")
            
            # Pause this DAG automatically in the Airflow metadata database
            with create_session() as session:
                dag_model = session.query(DagModel).filter(DagModel.dag_id == dag.dag_id).first()
                if dag_model:
                    dag_model.is_paused = True
            
            # Skip the task execution so it clearly indicates no work was done
            raise AirflowSkipException("Finished processing all app IDs in the file.")

        print(f"Processing batch of {len(batch_to_process)} items (Index {start_index} to {start_index + len(batch_to_process)})...")

        # Database Setup & Connection Management
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30.0)  # 30s timeout prevents database lock errors
        cursor = conn.cursor()

        fails = 0
        successful_calls = 0
        current_last_id = last_fetched_id

        try:
            # Fetch loop
            for app in batch_to_process:
                appid = app["appid"]

                # Stop if fail limit reached
                if fails >= FAIL_LIMIT:
                    print(f"Aborting batch early: reached fail limit of {FAIL_LIMIT}.")
                    break

                params = {"appids": appid, "format": "json"}
                url = f"{config.APP_DETAILS_URL}?{urlencode(params)}"

                try:
                    with urlopen(url, timeout=15) as response:
                        data = json.loads(response.read().decode("utf-8"))

                    data_str = json.dumps(data)
                    cursor.execute(
                        "INSERT OR REPLACE INTO records (id, data) VALUES (?, ?)",
                        (appid, data_str)
                    )

                    successful_calls += 1
                    current_last_id = appid
                    time.sleep(PAUSE_BETWEEN_CALLS)

                except HTTPError as e:
                    if e.code == 429:
                        print(f"HTTP 429 (Rate Limited) encountered at appid {appid}. Stopping batch immediately.")
                        break
                    
                    fails += 1
                    print(f"HTTP Error {e.code} for appid {appid}")
                    _log_failed_appid(appid)

                except Exception as e:
                    fails += 1
                    print(f"Failed fetching appid {appid}: {e}")
                    _log_failed_appid(appid)

            # Commit DB changes
            conn.commit()

        finally:
            # Always close connection even if task aborts unexpectedly
            conn.close()

        print(f"Batch completed. Successfully processed {successful_calls} apps.")
        
        if current_last_id != last_fetched_id:
            Variable.set("steam_last_appid_fetched", str(current_last_id))
            print(f"Updated Airflow Variable 'steam_last_appid_fetched' to {current_last_id}.")

    def _log_failed_appid(appid: int):
        FAILED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{appid}\n")

    fetch_steam_batch()