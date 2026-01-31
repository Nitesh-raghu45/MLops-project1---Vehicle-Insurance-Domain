from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URL"))
database = client["Proj_1"]

database["vehicle_insurance"].drop()

print("vehicle_insurance collection dropped successfully")
