#!/usr/bin/env python3
"""
This script will watch for new scan files anywhere in the 05-May
folder and uploads the experimental results to the database.

This should be run with:

python extract_qed_ip2_data.py
"""

import time
import re
import os
import pymongo
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from scipy.ndimage import gaussian_filter

# LabVIEW epoch (1904-01-01) to Unix epoch (1970-01-01) offset in seconds
LABVIEW_EPOCH_OFFSET = 2082844800

# data to be extracted
watched_folder = r'N:\data\Y2025\05-May'
experiment = 'staging_injector'
data_to_extract = {
    'Target focus': 'IP2-TC-ESP5 Position.Axis 1 Alias:Target focus',
    'Prepulse Delay': 'MANPAR-BELLA-General value10 Alias:prepulse_delay'
}

def extract_info_more_scan_file( path_to_scan_file ):
    # Open the scan file
    s_file = pd.read_csv(path_to_scan_file, sep='\t')

    # Extract the scan number from the file name
    m = re.search(r's(\d+)\.txt', path_to_scan_file)
    scan_number = int(m.groups(1)[0])

    # Check for required columns
    missing_column = False
    for _, value in data_to_extract.items():
        if value not in s_file.columns:
            print(f"Warning: Column '{value}' not found in {path_to_scan_file}")
            missing_column = True

    if not missing_column:
        # Loop over shots
        for i in range(len(s_file)):
            # Extract data for each shot
            data = {}
            data['experiment_flag'] = 1
            data['scan_number'] = scan_number
            data['shot_number'] = i
            # Convert LabVIEW timestamp to Unix timestamp by subtracting the offset
            labview_timestamp = s_file['DateTime Timestamp'].iloc[i]
            unix_timestamp = labview_timestamp - LABVIEW_EPOCH_OFFSET
            data['date'] = datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")

            # For now, use the date as a unique identifier:
            # if there is another shot in the database with the same date, skip it
            if collection.find_one({'date': data['date'], 'experiment_flag': 1}):
                print(f"Skipping shot {i} because it already exists in the database")
                continue

            # Extract required data
            for key, value in data_to_extract.items():
                data[key] = s_file[value].iloc[i]

            # Open corresponding camera image
            year, month, day = data['date'][:][:10].split('-')
            image_path = os.path.join( watched_folder,
                fr'{year[2:]}_{month}{day}\scans\Scan{scan_number:03d}\IP2-targetscatt\Scan{scan_number:03d}_IP2-targetscatt_{i+1:03d}.png')
            image = plt.imread(image_path).astype(np.float64)
            # Select pre-defined ROI, apply Gaussian filter
            imdata = gaussian_filter( image[100:400, 400:800], sigma=5 )
            # Coarse estimation of the harmonics energy: proportional to STD, with arbitrary constant
            data['harmonic_signal'] = 100*imdata.std()

            # Add to the database
            print('Uploading: ', data)
            collection.insert_one(data)

class MyHandler(FileSystemEventHandler):

    def __init__(self):
        # Compile the regex pattern once for efficiency
        # (Checks for new scan files, extracts scan number)
        self.pattern = re.compile(r'25_05\d\d\\analysis\\s\d+\.txt')

    def on_created(self, event):
        # Check if a new scan file has been created
        print(event.src_path)
        if self.pattern.search(event.src_path):
            print(f"New matching file created: {event.src_path}")
            extract_info_more_scan_file(event.src_path)


if __name__ == '__main__':

    # Open credential file for database
    with open('C:/Users/rlehe.BELLAAPPSERVER/Documents/db.profile') as f:
        db_profile = f.read()
    # Connect to the MongoDB database with read-only access
    db = pymongo.MongoClient(
        host="mongodb05.nersc.gov",
        username="bella_sf_admin",
        password=re.findall('SF_DB_ADMIN_PASSWORD=(.+)', db_profile)[0],
        authSource="bella_sf")["bella_sf"]
    collection = db[experiment]

    # Upload existing data in the database
    folders_to_upload = [
        os.path.join(watched_folder, r'25_0528\analysis'),
        os.path.join(watched_folder, r'25_0529\analysis'),
        os.path.join(watched_folder, r'25_0530\analysis')
    ]
    for folder_to_upload in folders_to_upload:
        for filename in os.listdir(folder_to_upload):
            print(filename)
            if re.match(r's\d+.txt',  filename ):
                extract_info_more_scan_file( os.path.join(folder_to_upload, filename) )

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
