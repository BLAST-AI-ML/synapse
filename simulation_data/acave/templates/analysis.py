#!/usr/bin/env python3

import os
import pymongo

# - Extract parameters from input script

# - Compute average wavelength

# - Log results to the database
db = pymongo.MongoClient(
    host="mongodb05.nersc.gov", 
    username="bella_sf_admin",
    password=os.getenv("SF_DB_ADMIN_PASSWORD"),
    authSource="bella_sf")["bella_sf"]

for collection in db.list_collections():
    print(collection['name'])

collection = db["acave"]



