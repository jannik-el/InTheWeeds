# %%
import cv2
import pytesseract
import pandas as pd
import re
from collections import Counter
import argparse
import os
import numpy as np
import pyodbc

CONNECTION_STRING = "Enter connection string here"

# %%

def py_arguments():
    parser = argparse.ArgumentParser(description='Process video file and extract frames where the target text is found')
    parser.add_argument('-vp', '--video_path', type=str, help='Path to the video file')
    parser.add_argument('-tn', '--table_name', type=str, help='Path to the excel file')
    parser.add_argument('-of', '--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('-tr', '--transect', type=str, help='Transect name')
    parser.add_argument('-v', '--verbose', default=False)
    args = parser.parse_args()
    return args

#  %%

def read_db(table_name):
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        query = f"SELECT * FROM dbo.{table_name}"
        cursor.execute(query)

        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        data = pd.DataFrame.from_records(rows, columns=columns)
        cursor.close()
        conn.close()
        # split datetime into date and time columns
        data["Date"] = data["Datetime"].dt.date
        data["Time"] = data["Datetime"].dt.time
        # sort by datetime
        data = data.sort_values(by=["Datetime"])
        return data
    except Exception as e:
        print("Error reading from database")
        print(e)
        return None
    
    
# %%
def format_ocr_text(ocr_text):
    regex_pattern = re.compile(r'(\d{2}-\d{2}-\d{4})|(Lat: \d+,\d+)|(\d{2}:\d{2}:\d{2})|(Long: \d+,\d+)')
    result = re.findall(regex_pattern, ocr_text)
    output = [re.sub(r'^\s*|\s*$', '', ' '.join(item)) for item in result]
    return output

def format_target_text(target_text, date_format="%d-%m-%Y", time_format="%H:%M:%S", gps_round=7, gps_commas=True):
    date = target_text["Date"]
    date = date.strftime(date_format)

    latitude = target_text["Latitude"]
    latitude = f'{float(latitude):.5f}'

    longitude = target_text["Longitude"]
    longitude = f'{float(longitude):.5f}'

    if gps_commas:
        latitude = str(latitude).replace(".", ",")
        longitude = str(longitude).replace(".", ",")

    time = target_text["Time"]
    time = time.strftime(time_format)

    return [date, f"Lat: {latitude}", time, f"Long: {longitude}"]

def process_video(data, video_path, output_folder, transect):
    date = format_target_text(data.iloc[0])[0]
    
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video file opened successfully
    if not video.isOpened():
        print("Error opening video file")
        exit()

    # Process video frames at the specified time interval
    frame_time = 0
    time_interval = 1

    idx_pointer = 0
    while video.isOpened():
        # Set the frame position to the current time
        video.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)

        # Read the current frame
        ret, frame = video.read()

        if not ret:
            break

        cropped_frame = frame[620:750, 0:600]
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Perform OCR on the grayscale frame
        ocr_text = pytesseract.image_to_string(thresh)
        ocr_tokens = format_ocr_text(ocr_text)

        if args.verbose: print("OCR Tokens: ", ocr_tokens)
                    
        idx = idx_pointer
        target_text_found = False

        loop_cnter = 0
        while not target_text_found and loop_cnter < 50:
            if idx >= len(data):
                break
            else:
                loop_cnter += 1
                # Iterate through rows of the DataFrame
                target_tokens = format_target_text(data.iloc[idx])
                # if args.verbose: print("Target Tokens: ", target_tokens)
                zostera_value = data.iloc[idx]["SliderCoverage"]

                if args.verbose: print(f"Target Tokens: {target_tokens}")

                # check how many tokens match
                token_counts = Counter(token for token in target_tokens if token in ocr_tokens)

                if args.verbose: print("Token Counts: ", token_counts)

                if len(token_counts) == 4:
                    current_time = data.iloc[idx]["Time"]
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"Datapoint {current_time} found and dataframe starting index set to: {idx}")
                    print(f"OCR Tokens:    {ocr_tokens}")
                    print(f"Target Tokens: {prnt_target_tokens}")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # Save the frame as an image
                    output_image_path = f'{output_folder}/{transect}_{date}_{current_time}_{zostera_value}.jpg'
                    cv2.imwrite(output_image_path, frame)
                    target_text_found = True
                    idx_pointer = idx+1
                            
                idx += 1

        # loop end time
        # calculate percent complete by looking at the current index in the dataframe and dividing it by the total number of rows
        percent_complete = round(idx_pointer/(len(data)+0.01)*100, 2)

        prnt_target_tokens = format_target_text(data.iloc[idx_pointer])
        if target_text_found == False:
            print(f"###############################################################")
            print(f"No datapoint found, {percent_complete}% complete")
            print(f"OCR Tokens:    {ocr_tokens}")
            print(f"Target Tokens: {prnt_target_tokens}")
            print(f"###############################################################")
            pass
                
        # Update the frame time
        frame_time += time_interval

    print("Done processing video.")

if __name__ == "__main__":
    args = py_arguments()
    data = read_db(args.table_name)

    # test that output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    process_video(data, args.video_path, args.output_folder, args.transect)