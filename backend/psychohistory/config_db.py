import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "psychohistory_db"
COLL_NAME = "predictions"

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
    coll = db[COLL_NAME]
    print("[DB] MongoDB unavailable, using in-memory mongomock")
