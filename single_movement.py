import serial
import time
import pandas as pd
import joblib
import tsfel
import matplotlib.pyplot as plt
from statistics import mode
import threading

def set_pwm(state):
    if state not in [0, 1]:
        raise ValueError("State must be 0 or 1")
    ser.write(f"{state}".encode())
    response = ser.readline().decode().strip()
    print(f"Teensy responded: {response}")

def trigger_fes(label):
    if label == 0:
        return
    def pwm_task():
        try:
            set_pwm(1)
            time.sleep(2)
            set_pwm(0)
        except Exception as e:
            print(f"Error occured: {e}")
    
    threading.Thread(target=pwm_task, daemon=True).start()

def extract_sliding_windows(window_size=500, stride=100):
    """Extract sliding windows from the data file."""
    df = pd.read_csv("5sec-rest-grasp-rest.csv", sep=",", header=None)
    df = df.iloc[:, :2]  # Keep only the first two columns
    windows = []
    num_rows = len(df)
    for start in range(0, num_rows - window_size + 1, stride):
        window_df = df.iloc[start: start + window_size].copy()
        windows.append(window_df)
    return windows

def main():
    STATIC_THRESHHOLD_VALUE = 5
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

    global_df = pd.read_csv("5sec-rest-grasp-rest.csv", sep=",", header=None).iloc[:, :2]
    global_min0, global_max0 = global_df.iloc[:, 0].min(), global_df.iloc[:, 0].max()
    global_min1, global_max1 = global_df.iloc[:, 1].min(), global_df.iloc[:, 1].max()

    # Create Matplotlib figure and axes
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.4)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.move(0, 0)
        manager.window.resize(960, 1080)
    except Exception as e:
        pass

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
    # plt.ion()
    # fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    # fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=0.4)
    # manager = plt.get_current_fig_manager()
    # try:
    #     manager.window.move(0, 0)
    #     manager.window.resize(960, 1080)
    # except Exception:
    #     pass

    # line0, = axs[0].plot([], [], color="blue")
    # axs[0].set_title("Channel 1", fontsize=14, fontweight='bold')
    # axs[0].set_ylabel("Signal Voltage (MicroVolts)", fontsize=12, fontweight='bold')
    # axs[0].set_xlim(0, 500)
    # axs[0].set_ylim(global_min0, global_max0)

    # line1, = axs[1].plot([], [], color="green")
    # axs[1].set_title("Channel 2", fontsize=14, fontweight='bold')
    # axs[1].set_xlabel("Samples over Time (Samples)", fontsize=12, fontweight='bold')
    # axs[1].set_ylabel("Signal Voltage (MicroVolts)", fontsize=12, fontweight='bold')
    # axs[1].set_xlim(0, 500)
    # axs[1].set_ylim(global_min1, global_max1)

    for window in all_windows:
        window.columns = [f"channel_{i}" for i in range(window.shape[1])]
        
        extracted_features = tsfel.time_series_features_extractor(cfg, window, fs=1000, verbose=0)
        extracted_features.columns = [
            name.replace("channel1", "channel_0").replace("channel2", "channel_1")
            for name in extracted_features.columns
        ]
        extracted_features = extracted_features.reindex(columns=expected_feature_names, fill_value=0)
        
        prediction = rf_model.predict(extracted_features)[0]
        output.append(prediction)

        if len(output) > STATIC_THRESHHOLD_VALUE:
            output.pop(0)

        if len(output) == STATIC_THRESHHOLD_VALUE:
            label = mode(output)
            if label != previous:
                previous = label
                print(f"Window predicted: {label}")
                if label == 0:
                    fig.suptitle(f"No FES Delivered For Movement: {movement_encoding[label]}", fontsize=16, fontweight='bold')
                else:
                    fig.suptitle(f"FES Delivered For Movement: {movement_encoding[label]}", fontsize=16, fontweight='bold')
                    trigger_fes(label)
                    print(label)

        x = window.index
        y0 = window["channel_0"].values
        y1 = window["channel_1"].values

        line0.set_data(x, y0)
        line1.set_data(x, y1)
        axs[0].set_xlim(x.min(), x.max())
        axs[1].set_xlim(x.min(), x.max())

        plt.draw()
        plt.pause(0.1)

    plt.close(fig)

if __name__ == "__main__":
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)  # Give the Teensy time to reset after serial opens

    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        ser.close()
        print("Serial connection closed.")
