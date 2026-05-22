import tempfile
import os
from database import init_db, save_result, fetch_all

def test_db_save_and_fetch():
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    try:
        init_db(db_path)
        result = {
            'soil_type': 'Test Soil',
            'moisture': 'Low',
            'compaction': 'Low',
            'dryness': 'Low',
            'health_score': 90,
            'recommendation': 'None'
        }
        metadata = {
            'sample_id': 'test123',
            'location': 'lab',
            'notes': 'unit test',
            'uploaded_at': '20260101T000000Z',
            'filename': 'test.pdf',
            'waveform': None,
            'feature_file': None
        }
        save_result(result, metadata, db_path=db_path)
        rows = fetch_all(db_path=db_path)
        assert len(rows) == 1
        assert rows[0]['sample_id'] == 'test123'
        assert rows[0]['soil_type'] == 'Test Soil'
    finally:
        try:
            os.remove(db_path)
        except Exception:
            pass
