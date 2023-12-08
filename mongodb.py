import json
from pymongo import MongoClient

class MongoDBManager:
    def __init__(self, url, db_name):
        self.client = MongoClient(url)
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)
        self.db = self.client[db_name]

    def validate_json(self, json_data):
        required_keys = ["interest", "preferred_callback_times", "demo", "demo_schedule", "budget", 
                         "decision_makers_and_process", "pain_points", "competitors", "timeline", 
                         "previous_solutions_experience"]
        return all(key in json_data for key in required_keys)

    def insert_json_into_collection(self, collection_name, json_data):
        if self.validate_json(json_data):
            collection = self.db[collection_name]
            collection.insert_one(json_data)
            return True
        else:
            print("Invalid JSON data. Please make sure it contains all the required keys.")
            return False