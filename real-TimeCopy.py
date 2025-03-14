import joblib
import tsfel
import pandas as pd
import numpy as np
import glob
from collections import Counter, defaultdict
from statistics import mode
import matplotlib.pyplot as plt  
import cv2
import threading

def extract_sliding_windows(window_size=500, stride=100):
    # Read the file into a DataFrame
    # df = pd.read_csv("0-8_half_droppedClass.txt", sep="\t", header=None)
    df = pd.read_csv("0-8_half_droppedClass.csv", sep=",", header=None)

    # Keep only 2 out of 3 columns
    df = df.iloc[:, :2]  # This selects only columns 0 and 1

    # Create sliding windows as a list of DataFrames
    windows = []
    num_rows = len(df)
    for start in range(0, num_rows - window_size + 1, stride):
        # Extract a slice (window) of the DataFrame
        window_df = df.iloc[start : start + window_size].copy()
        windows.append(window_df)
    
    return windows

#ADD ONE BIG LOOP TO MAKE IT ALL AT ONCE
#SPLIT THE SCREEN TO BE LEFT AND RIGHT

def playVideo(label):

    # if label == 0:  
    #     video_path = "rest.MOV"
    # elif label == 1:
    #     video_path = "grasp.MOV"
    # elif label == 2:
    #     video_path = "release.MOV"
    # elif label == 3:
    #     video_path = "curl.MOV"
    # elif label == 4:
    #     video_path = "extend.MOV"
    # elif label == 5:
    #     video_path = "pinch.MOV"
    # elif label == 6:
    #     video_path = "jaw.MOV"
    # elif label == 7:
    #     video_path = "thumb down.MOV"
    # elif label == 8:
    #     video_path = "thumb up.MOV"

    # Create a VideoCapture object
    # cap = cv2.VideoCapture(video_path)
    

    # video_paths = {
    #                 0: "rest.MOV",
    #                 1: "grasp.MOV",
    #                 2: "release.MOV",
    #                 3: "curl.MOV",
    #                 4: "extend.MOV",
    #                 5: "pinch.MOV",
    #                 6: "jaw.MOV",
    #                 7: "thumb down.MOV",
    #                 8: "thumb up.MOV"
    #               }       
    video_paths = {
                    1: "GRASP_cut.mp4",
                    2: "RELEASE_cut.mp4",
                    3: "CURL_cut.mp4",
                    4: "EXTEND_cut.mp4",
                    5: "PINCH_cut.mp4",
                    6: "JAW_cut.mp4",
                    7: "THUMBDOWN_cut.mp4",
                    8: "THUMBUP_cut.mp4"
                  }  
    
    # video_path = video_paths.get(label)
    
    # # Create a named window and move it to the right side of the screen.
    # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # # Change the (x, y) values below as needed.
    # cv2.moveWindow('Video', 0, 480)  # x=800 positions it on the right side, y=0 is the top.
    
    # cap = cv2.VideoCapture(video_path)
    

    video_path = video_paths.get(label)
    
    # Create a named window and position it for bottom right.
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    
    # Fixed dimensions (example values)
    screen_width = 1920
    screen_height = 1080
    video_window_width = 640
    video_window_height = 480
    
    # Calculate position for bottom right
    x_position = screen_width - video_window_width
    y_position = screen_height - video_window_height
    cv2.moveWindow('Video', x_position, y_position)
    
    cap = cv2.VideoCapture(video_path)


    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # Read until video is completed or the user presses 'q'
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Break the loop if there are no frames to read
                break

            # Display the frame in a window named 'Video'
            cv2.imshow('Video', frame)

            # Press 'q' on the keyboard to exit early
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release the video capture object and close display window
        cap.release()
        cv2.destroyAllWindows()

    


def main():
    STATIC_THRESHHOLD_VALUE = 18
    all_windows = extract_sliding_windows()
    output = []
    previous = -1
    
    movement_encoding = {
                            0 : "Rest", 
                            1 : "Grasp",
                            2 : "Release", 
                            3 : "Curl", 
                            4 : "Extend", 
                            5 : "Pinch", 
                            6 : "Jaw", 
                            7 : "Thumb Down", 
                            8 : "Thumb Up"
                        }
    
    rf_model = joblib.load("rf_model.pkl")
    cfg = tsfel.get_features_by_domain()
    expected_feature_names = rf_model.feature_names_in_  # Names used in training
    
    # Read the entire dataset to compute global y-axis limits for each channel.
    #global_df = pd.read_csv("0-8_half_droppedClass.txt", sep="\t", header=None).iloc[:, :2]
    global_df = pd.read_csv("0-8_half_droppedClass.csv", sep=",", header=None).iloc[:, :2]
    global_min0, global_max0 = global_df.iloc[:, 0].min(), global_df.iloc[:, 0].max()
    global_min1, global_max1 = global_df.iloc[:, 1].min(), global_df.iloc[:, 1].max()

    # Set up a single figure with two subplots in interactive mode
    plt.ion()  # Turn on interactive mode for seamless updates
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    # Position the Matplotlib window on the left side of the screen
    manager = plt.get_current_fig_manager()
    # manager.window.move(0, 0)
    manager.window.geometry("+0+0")


    # Initialize plot lines and set fixed axis limits
    line0, = axs[0].plot([], [], color="blue")
    axs[0].set_title("Channel 1")
    axs[0].set_ylabel("Signal Voltage (MicroVolts)")
    axs[0].set_xlim(0, 500)  # Window size is 500 samples
    axs[0].set_ylim(global_min0, global_max0)  # Fixed y-axis limits for channel 1
    
    line1, = axs[1].plot([], [], color="green")
    axs[1].set_title("Channel 2")
    axs[1].set_xlabel("Samples over Time (Samples)")
    axs[1].set_ylabel("Signal Voltage (MicroVolts)")
    axs[1].set_xlim(0, 500)
    axs[1].set_ylim(global_min1, global_max1)  # Fixed y-axis limits for channel 2

    # Loop over each window
    for window in all_windows:
        window.columns = [f"channel_{i}" for i in range(window.shape[1])]
        
        # Extract TSFEL features
        extracted_features = tsfel.time_series_features_extractor(cfg, window, fs=1000, verbose=0)
        #start plotting
        extracted_features.columns = [name.replace("channel1", "channel_0").replace("channel2", "channel_1")
                                       for name in extracted_features.columns]
        extracted_features = extracted_features.reindex(columns=expected_feature_names, fill_value=0)
        

        # Perform classification
        prediction = rf_model.predict(extracted_features)[0]
        output.append(prediction)        

        
        
        # if len(output) == STATIC_THRESHHOLD_VALUE:
        #     label = mode(output)
        #     output = []
        #     if label != previous:
        #         previous = label
        
        #         print(f"Window predicted: {label}")

        #         if (label == 0):
        #             fig.suptitle(f"No FES Delivered For Movement: {movement_encoding[label]}")
        #         else:
        #             fig.suptitle(f"FES Delivered For Movement: {movement_encoding[label]}")
                    
        #         playVideo(label)
        


        # Append new prediction to the sliding window
        if len(output) > STATIC_THRESHHOLD_VALUE:
            
            output.pop(0)  # Remove the oldest prediction
            

        # When the sliding window is full, calculate the mode
        if len(output) == STATIC_THRESHHOLD_VALUE:
            label = mode(output)
            if label != previous:
                previous = label
                print(f"Window predicted: {label}")
                if label == 0:
                    fig.suptitle(f"No FES Delivered For Movement: {movement_encoding[label]}") #MAKE THE LABELS BOLDED AND COLORED
                else:
                    fig.suptitle(f"FES Delivered For Movement: {movement_encoding[label]}")
                if label != 0:
                    video_thread = threading.Thread(target=playVideo, args=(label,), daemon=True)
                    video_thread.start()


        
        # Update the plot with new data for both channels
        # x = np.arange(window.shape[0])
        # y0 = window["channel_0"].values
        # y1 = window["channel_1"].values
        
        # line0.set_data(x, y0)
        # line1.set_data(x, y1)
        
        # plt.draw()         # Redraw the current figure
        # plt.pause(0.1)     # Short pause to simulate real-time update
        
        # Use the DataFrame's index as the x values
        

        x = window.index  
        y0 = window["channel_0"].values
        y1 = window["channel_1"].values

        line0.set_data(x, y0)
        line1.set_data(x, y1)

        # Update x-axis limits to reflect the current window's indices
        axs[0].set_xlim(x.min(), x.max())
        axs[1].set_xlim(x.min(), x.max())

        plt.draw()
        plt.pause(0.1)

    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Display the final figure

if __name__ == "__main__":
    main()


