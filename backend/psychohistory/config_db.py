import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "psychohistory_db"
COLL_NAME = "predictions"
DATA_FILE = Path(__file__).resolve().parent.parent / ".data" / "predictions.json"
MAX_PERSISTED = 200


class _PersistentCollection:
    """mongomock collection backed by a JSON file so predictions survive restarts.

    Only the fallback path uses this. It delegates every operation to an inner
    mongomock collection and rewrites the JSON snapshot after each insert.
    """

    def __init__(self, inner, path: Path):
        self._inner = inner
        self._path = path
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                docs = json.loads(self._path.read_text())
                if docs:
                    self._inner.insert_many(docs)
                    print(f"[DB] Restored {len(docs)} predictions from {self._path.name}")
        except Exception as e:
            print(f"[DB] Restore error: {e}")

    def _dump(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            docs = list(self._inner.find({}, {"_id": 0}))
            docs.sort(key=lambda d: d.get("timestamp", 0))
            self._path.write_text(json.dumps(docs[-MAX_PERSISTED:]))
        except Exception as e:
            print(f"[DB] Persist error: {e}")

    def insert_one(self, document):
        result = self._inner.insert_one(document)
        self._dump()
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


try:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.admin.command("ping")
    db = client[DB_NAME]
    coll = db[COLL_NAME]
    print("[DB] Connected to native MongoDB")
except Exception:
    import mongomock
    client = mongomock.MongoClient()
    db = client[DB_NAME]
    coll = _PersistentCollection(db[COLL_NAME], DATA_FILE)
    print("[DB] MongoDB unavailable, using file-backed store")
