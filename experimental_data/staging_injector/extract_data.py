#!/usr/bin/env python3
"""
This script will watch for new scan files anywhere in the 08-Aug
folder and uploads the experimental results to the database.

This should be run with:

python extract_data.py
"""

import time
import re
import os
import pymongo
from datetime import datetime
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np

# LabVIEW epoch (1904-01-01) to Unix epoch (1970-01-01) offset in seconds
LABVIEW_EPOCH_OFFSET = 2082844800

# data to be extracted
watched_folder = r"N:\data\Y2025\08-Aug"
watched_pattern = r"25_08"
experiment = "staging_injector"
data_to_extract = {
    "Amplifier 3 [J]": "EM-LB-2 Reading.Channel 2 Alias:Amplifier 3 [J]",
    "HEX-PL1-1 xpos": "HEX-PL1-1 xpos",
    "Valve01": "MANPAR-BELLA-ValveN2Frac value1 Alias:Valve01",
    "Cap Up (torr)": "GAUGE-PL1-CapPressure pressure Alias:Cap Up (torr)",
    "Cap downstream (torr)": "GAUGE-PL1-CapPressure pressure2 Alias:Cap downstream (torr)",
    "Beam mean energy [GeV]": "MGS meanMomentum_GeV/c",
    "EBeamPrf charge [pC]": "EBeamPrf charge [pC]",
    "EBeamPrf fwhm div x [mrad]": "EBeamPrf fwhm div x [mrad]",
    "SPEC-AA-Hamamastsu lambda_b": "SPEC-AA-Hamamastsu lambda_b",
    "SPEC-AA-Hamamastsu lambda_r": "SPEC-AA-Hamamastsu lambda_r",
}
unavailable_data = [
    "Beam energy spread [%]",
]


def extract_info_more_scan_file(path_to_scan_file):
    # Open the scan file
    s_file = pd.read_csv(path_to_scan_file, sep="\t")

    # Extract the scan number from the file name
    m = re.search(r"s(\d+)\.txt", path_to_scan_file)
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

            # Check is this shot should be uploaded
            # (charge should be higher than 10 pC ; beam should be within 0.5 mrad of the center)
            skip_shot = False
            if s_file["EBeamPrf charge [pC]"].iloc[i] < 10:
                skip_shot = True
            elif abs(s_file["EBeamPrf mean angle x [mrad]"].iloc[i]) > 0.5:
                skip_shot = True
            elif abs(s_file["EBeamPrf mean angle y [mrad]"].iloc[i]) > 0.5:
                skip_shot = True
            elif ('25_0827' in path_to_scan_file) and ((scan_number < 19) or (scan_number==21)):
                skip_shot = True
            if skip_shot:
                continue

            # Extract data for each shot
            data = {}
            data["experiment_flag"] = 1
            data["scan_number"] = scan_number
            data["shot_number"] = i
            # Convert LabVIEW timestamp to Unix timestamp by subtracting the offset
            labview_timestamp = s_file["DateTime Timestamp"].iloc[i]
            unix_timestamp = labview_timestamp - LABVIEW_EPOCH_OFFSET
            data["date"] = datetime.fromtimestamp(unix_timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # For now, use the date as a unique identifier:
            # if there is another shot in the database with the same date, skip it
            if collection.find_one({"date": data["date"], "experiment_flag": 1}):
                print(f"Skipping shot {i} because it already exists in the database")
                continue

            # Set unavailable data to NaN
            for key in unavailable_data:
                data[key] = np.nan

            # Extract required data
            for key, value in data_to_extract.items():
                data[key] = np.float64(s_file[value].iloc[i])

            # Add to the database
            print("Uploading: ", data)
            collection.insert_one(data)


class MyHandler(FileSystemEventHandler):
    def __init__(self):
        # Compile the regex pattern once for efficiency
        # (Checks for new scan files, extracts scan number)
        self.pattern = re.compile(watched_pattern + r"\d\d\\analysis\\s\d+\.txt")

    def on_created(self, event):
        # Check if a new scan file has been created
        print(event.src_path)
        if self.pattern.search(event.src_path):
            print(f"New matching file created: {event.src_path}")
            extract_info_more_scan_file(event.src_path)


if __name__ == "__main__":
    # Open credential file for database
    with open("C:/Users/rlehe.BELLAAPPSERVER/Documents/db.profile") as f:
        db_profile = f.read()
    # Connect to the MongoDB database with read-write access
    db = pymongo.MongoClient(
        host="mongodb05.nersc.gov",
        username="bella_sf_admin",
        password=re.findall("SF_DB_ADMIN_PASSWORD=(.+)", db_profile)[0],
        authSource="bella_sf",
    )["bella_sf"]
    collection = db[experiment]

    # Upload existing data in the database
    folders_to_upload = [
        os.path.join(watched_folder, r"25_0827\analysis"),
    ]
    for folder_to_upload in folders_to_upload:
        for filename in os.listdir(folder_to_upload):
            print(filename)
            if re.match(r"s\d+.txt", filename):
                extract_info_more_scan_file(os.path.join(folder_to_upload, filename))

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