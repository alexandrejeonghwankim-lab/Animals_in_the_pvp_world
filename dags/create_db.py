import sqlite3
from pathlib import Path
from pendulum import datetime
from airflow.sdk import DAG, task

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "steam_raw_data.db"

with DAG(
    dag_id="00_init_database",
    start_date=datetime(2026, 9, 1),
    schedule=None,  # Guarantees no automated runs
    catchup=False,
    tags=["setup", "manual-only"],
) as dag:

    @task
    def create_database_tables():
        print(f"Connecting to database at: {DB_PATH}")

        # Ensure the target /data folder actually exists before creating the file
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY NOT NULL,
            data TEXT NOT NULL,
            date_collected TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        conn.close()
        print("Database and table created successfully.")

    create_database_tables()