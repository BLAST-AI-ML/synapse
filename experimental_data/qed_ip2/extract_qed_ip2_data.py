#!/usr/bin/env python3
"""
This script will watch for new scan files anywhere in the 05-May
folder and uploads the experimental results to the database.

This should be run with:

python extract_qed_ip2_data.py
"""

import time
import re
import pymongo
from datetime import datetime
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# LabVIEW epoch (1904-01-01) to Unix epoch (1970-01-01) offset in seconds
LABVIEW_EPOCH_OFFSET = 2082844800

# data to be extracted
watched_folder = r'N:\data\Y2025\05-May'
experiment = 'qed_ip2'
data_to_extract = {
    'Target focus': 'IP2-TC-ESP5 Position.Axis 1 Alias:Target focus',
    'Prepulse Delay': 'IP2-Zaber-2 Position.Ch3 Alias:Prepulse Delay'
}

class MyHandler(FileSystemEventHandler):

    def __init__(self):
        # Compile the regex pattern once for efficiency (Checks for new scan files)
        self.pattern = re.compile(r'25_05\d\d\\analysis\\s\d+\.txt')

    def on_created(self, event):
        # Check if a new scan file has been created
        if self.pattern.search(event.src_path):
            print(f"New matching file created: {event.src_path}")

            # In that case, open the s file
            s_file = pd.read_csv(event.src_path, sep='\t')

            # Check for required columns
            missing_column = False
            for _, value in data_to_extract.items():
                if value not in s_file.columns:
                    print(f"Warning: Column '{value}' not found in {event.src_path}")
                    missing_column = True

            if not missing_column:
                # Loop over shots
                for i in range(len(s_file)):
                    # Extract data for each shot
                    data = {}
                    data['experiment_flag'] = 1
                    # Convert LabVIEW timestamp to Unix timestamp by subtracting the offset
                    labview_timestamp = s_file['DateTime Timestamp'].iloc[i]
                    unix_timestamp = labview_timestamp - LABVIEW_EPOCH_OFFSET
                    data['date'] = datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    # Extract required data
                    for key, value in data_to_extract.items():
                        data[key] = s_file[value].iloc[i]

                    # TODO: Open corresponding image ; integrate energy
                    data['energy_in_harmonics'] = 0

                    # Add to the database
                    collection.insert_one(data)

if __name__ == '__main__':

    # Open credential file for database
    with open('C:/Users/rlehe.BELLAAPPSERVER/Documents/db.profile') as f:
#    with open('/Users/rlehe/db.profile') as f:
        db_profile = f.read()
    # Connect to the MongoDB database with read-only access
    db = pymongo.MongoClient(
        host="mongodb05.nersc.gov",
        username="bella_sf_admin",
        password=re.findall('SF_DB_ADMIN_PASSWORD=(.+)', db_profile)[0],
        authSource="bella_sf")["bella_sf"]
    collection = db[experiment]

    # Create an observer and handler
    observer = Observer()
    event_handler = MyHandler()

    # Schedule the observer to watch the directory
    observer.schedule(event_handler, path=watched_folder, recursive=True)
    observer.start()

    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()