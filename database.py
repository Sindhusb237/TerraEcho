import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "terraecho.db")


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT,
            location TEXT,
            notes TEXT,
            uploaded_at TEXT,
            filename TEXT,
            soil_type TEXT,
            moisture TEXT,
            compaction TEXT,
            dryness TEXT,
            health_score INTEGER,
            recommendation TEXT
        )
        """
    )
    conn.commit()
    # ensure new optional columns exist (safe for existing DB)
    cur.execute("PRAGMA table_info(samples)")
    cols = [r[1] for r in cur.fetchall()]
    if 'waveform_file' not in cols:
        try:
            cur.execute("ALTER TABLE samples ADD COLUMN waveform_file TEXT")
        except Exception:
            pass
    if 'feature_file' not in cols:
        try:
            cur.execute("ALTER TABLE samples ADD COLUMN feature_file TEXT")
        except Exception:
            pass

    conn.commit()
    conn.close()


def save_result(result: dict, metadata: dict, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO samples (
            sample_id, location, notes, uploaded_at, filename,
            soil_type, moisture, compaction, dryness, health_score, recommendation,
            waveform_file, feature_file
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.get("sample_id"),
            metadata.get("location"),
            metadata.get("notes"),
            metadata.get("uploaded_at"),
            metadata.get("filename"),
            result.get("soil_type"),
            result.get("moisture"),
            result.get("compaction"),
            result.get("dryness"),
            result.get("health_score"),
            result.get("recommendation"),
            metadata.get("waveform"),
            metadata.get("feature_file"),
        ),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()


def fetch_all(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sample_id, location, notes, uploaded_at, filename, soil_type, moisture, compaction, dryness, health_score, recommendation, waveform_file, feature_file FROM samples ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    conn.close()

    keys = ["sample_id", "location", "notes", "uploaded_at", "filename", "soil_type", "moisture", "compaction", "dryness", "health_score", "recommendation", "waveform", "feature_file"]
    results = [dict(zip(keys, row)) for row in rows]
    return results
