import os

from dotenv import load_dotenv
from mongoengine import connect

load_dotenv(verbose=True)

MONGO_DB = os.getenv("MONGO_DB")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT"))
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASS = os.getenv("MONGO_PASS")
AUTHENTICATION_SOURCE = os.getenv("AUTHENTICATION_SOURCE")

res = connect(db=MONGO_DB, host=MONGO_HOST, port=MONGO_PORT,
              username=MONGO_USER, password=MONGO_PASS,
              authentication_source=AUTHENTICATION_SOURCE)
print(res)
