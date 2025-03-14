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
    df = pd.read_csv("0-8_half_droppedClass.csv", sep=",", header=None)
    # Keep only the first 2 columns
    df = df.iloc[:, :2]
    # Create sliding windows as a list of DataFrames
    windows = []
    num_rows = len(df)
    for start in range(0, num_rows - window_size + 1, stride):
        window_df = df.iloc[start: start + window_size].copy()
        windows.append(window_df)
    return windows

def playVideo(label):
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
    
    video_path = video_paths.get(label)
    
    # Create a named window with autosize so that the video remains at its original dimensions.
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    # Move the video window to the right half (x=960, y=0)
    cv2.moveWindow('Video', 960, 0)
    # Set the video window to topmost so it's not hidden.
    cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    STATIC_THRESHHOLD_VALUE = 18
    all_windows = extract_sliding_windows()
    output = []
    previous = -1
    
    movement_encoding = {
        0: "Rest", 
        1: "Grasp",
        2: "Release", 
        3: "Curl", 
        4: "Extend", 
        5: "Pinch", 
        6: "Jaw", 
        7: "Thumb Down", 
        8: "Thumb Up"
    }
    
    rf_model = joblib.load("rf_model.pkl")
    cfg = tsfel.get_features_by_domain()
    expected_feature_names = rf_model.feature_names_in_
    
    # Compute global y-axis limits for both channels.
    global_df = pd.read_csv("0-8_half_droppedClass.csv", sep=",", header=None).iloc[:, :2]
    global_min0, global_max0 = global_df.iloc[:, 0].min(), global_df.iloc[:, 0].max()
    global_min1, global_max1 = global_df.iloc[:, 1].min(), global_df.iloc[:, 1].max()

    # Create the Matplotlib figure.
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.4)
    
    # Reposition the Matplotlib window to cover the left half of the screen.
    manager = plt.get_current_fig_manager()
    try:
        manager.window.move(0, 0)
        manager.window.resize(960, 1080)
    except Exception as e:
        # If repositioning isn't available on your platform, it will remain at default.
        pass

    # Initialize plot lines and set fixed axis limits with bolded titles and labels.
    line0, = axs[0].plot([], [], color="blue")
    axs[0].set_title("Channel 1", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Signal Voltage (MicroVolts)", fontsize=12, fontweight='bold')
    axs[0].set_xlim(0, 500)
    axs[0].set_ylim(global_min0, global_max0)
    
    line1, = axs[1].plot([], [], color="green")
    axs[1].set_title("Channel 2", fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Samples over Time (Samples)", fontsize=12, fontweight='bold')
    axs[1].set_ylabel("Signal Voltage (MicroVolts)", fontsize=12, fontweight='bold')
    axs[1].set_xlim(0, 500)
    axs[1].set_ylim(global_min1, global_max1)

    # Loop over each sliding window and update the graph.
    for window in all_windows:
        window.columns = [f"channel_{i}" for i in range(window.shape[1])]
        
        # Extract TSFEL features.
        extracted_features = tsfel.time_series_features_extractor(cfg, window, fs=1000, verbose=0)
        extracted_features.columns = [
            name.replace("channel1", "channel_0").replace("channel2", "channel_1")
            for name in extracted_features.columns
        ]
        extracted_features = extracted_features.reindex(columns=expected_feature_names, fill_value=0)
        
        # Perform classification.
        prediction = rf_model.predict(extracted_features)[0]
        output.append(prediction)
        if len(output) > STATIC_THRESHHOLD_VALUE:
            output.pop(0)
        
        # When the sliding window is full, calculate the mode and update if changed.
        if len(output) == STATIC_THRESHHOLD_VALUE:
            label = mode(output)
            if label != previous:
                previous = label
                print(f"Window predicted: {label}")
                # Set the suptitle (FES movement text) with larger, bold text.
                if label == 0:
                    fig.suptitle(
                        f"No FES Delivered For Movement: {movement_encoding[label]}",
                        fontsize=16, fontweight='bold'
                    )
                else:
                    fig.suptitle(
                        f"FES Delivered For Movement: {movement_encoding[label]}",
                        fontsize=16, fontweight='bold'
                    )
                if label != 0:
                    video_thread = threading.Thread(target=playVideo, args=(label,), daemon=True)
                    video_thread.start()

        x = window.index  
        y0 = window["channel_0"].values
        y1 = window["channel_1"].values

        line0.set_data(x, y0)
        line1.set_data(x, y1)
        axs[0].set_xlim(x.min(), x.max())
        axs[1].set_xlim(x.min(), x.max())

        plt.draw()
        plt.pause(0.1)

    # Instead of plt.show(), automatically close the plot when done.
    plt.close(fig)

if __name__ == "__main__":
    while True:
        main()
