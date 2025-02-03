# Step 1: Load the Data
# We will begin by loading the text file containing the EMG data and audio trigger. The data consists of three columns: two for the EMG signals (Channel 1 and Channel 2) and one for the audio trigger (Channel 3).
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the text file data
file_path = 'Huncho_Curl_Extend_1_30.txt'
data = pd.read_csv(file_path, delimiter='\t', header=None)

# Extract EMG channels and audio trigger
emg_data = data.iloc[:, 0:2]  # EMG Channels (Channel 1 and Channel 2)
audio_trigger = data.iloc[:, 2]  # Audio Trigger (Channel 3)

# Step 2: Set Up Parameters and Threshold
# Now, we set up the sampling frequency (based on your earlier context of 0.05 ms per sample) and a threshold for noise detection. The threshold will help us identify points in the audio trigger signal where significant noise events occur.

# Sampling frequency (adjust based on your actual sample rate)
sampling_frequency = 10000  # Assuming 0.05 ms per sample from earlier data context
duration_samples = int(10 * sampling_frequency)  # 10-second window size

# Set a threshold for noise detection
threshold = np.mean(audio_trigger) + 2 * np.std(audio_trigger)

# Step 3: Detect Noise Events
# We will now detect points where the audio trigger signal exceeds the threshold, indicating noise events. This is done by comparing each value in the audio trigger to the threshold and marking the points where the signal is above it.
# Detect points where the signal is above the threshold
signal_above_threshold = audio_trigger > threshold

# Step 4: Merge Close Noise Events
# After detecting individual noise events, we merge consecutive events that are too close to each other. This helps reduce noise and false positives. The merging process takes into account a minimum gap between noise events (in samples).

# Define a minimum gap between noise events (in samples)
min_gap = 50  # Adjust this value as needed

# Initialize a list to store the merged noise events
merged_noise_events = []
start = None

# Iterate through the signal_above_threshold and detect/merge noise events
for i, above in enumerate(signal_above_threshold):
    if above and start is None:
        start = i  # Mark the start of a noise event
    elif not above and start is not None:
        if i - start > 10:  # Minimum event duration to filter out short spikes
            # Check if the current event is too close to the previous event
            if len(merged_noise_events) > 0:
                prev_start, prev_end = merged_noise_events[-1]
                if start - prev_end <= min_gap:  # If the gap is small, merge the events
                    merged_noise_events[-1] = (prev_start, i)  # Merge events
                else:
                    merged_noise_events.append((start, i))  # Add as new event
            else:
                merged_noise_events.append((start, i))  # Add first event
        start = None  # Reset the start for the next potential event

# Step 5: Calculate Samples Between Events
# Next, we calculate the number of samples between each pair of merged noise events. This will help us understand the interval between detected movements.
# Calculate the number of samples between each pair of merged events
samples_between_events = []
for i in range(1, len(merged_noise_events)):
    prev_end = merged_noise_events[i - 1][1]
    current_start = merged_noise_events[i][0]
    
    # Calculate the difference between the start of the current event and the end of the previous one
    samples_between_events.append(current_start - prev_end)

print("Samples between events:", samples_between_events)

# Calculate the average number of samples between noise events
avg_samples_between_events = np.mean(samples_between_events)
print(f"Average samples between events: {avg_samples_between_events}")

# Step 6: Label the Movements
# We will now label the movements based on the merged noise events. Movement 1 and Movement 2 will alternate, and the rest periods will be filled in as needed.
# Now, let's label the movements based on merged noise events
movement_labels = ["Rest"] * len(audio_trigger)

# Label the movements (alternating between Movement 1 and Movement 2)
movement_types = ["Movement 1", "Movement 2"]
for idx, (start, end) in enumerate(merged_noise_events):
    movement_label = movement_types[idx % 2]  # Alternate between Movement 1 and Movement 2
    movement_labels[start:end] = [movement_label] * (end - start)

# For the last "Movement 2", extend the label based on the average interval and switch to "Rest"
last_movement_end = merged_noise_events[-1][1] + avg_samples_between_events
last_movement_end = min(last_movement_end, len(audio_trigger))
movement_labels[merged_noise_events[-1][1]:int(last_movement_end)] = ["Movement 2"] * (int(last_movement_end) - merged_noise_events[-1][1])

# Switch to "Rest" after the last "Movement 2"
movement_labels[int(last_movement_end):] = ["Rest"] * (len(audio_trigger) - int(last_movement_end))

# Step 7: Save the Labeled Data
# Finally, we save the labeled data (including both the EMG channels and the movement labels) into a CSV file for further analysis.
# Create a DataFrame for the labeled data
labeled_data = pd.DataFrame(emg_data, columns=["Channel 1", "Channel 2"])
labeled_data["Audio Trigger"] = audio_trigger
labeled_data["Movement Label"] = movement_labels

# Save the labeled data to a CSV file
labeled_data.to_csv("labeled_data.csv", index=False)

# Step 8: Visualize the Data
# Now, let’s plot the EMG data for Channel 1 and Channel 2, highlighting the regions corresponding to Movement 1, Movement 2, and Rest periods. We will also plot the audio trigger with the detected noise events.
# Plot the EMG channels with movement labels
plt.figure(figsize=(12, 10))

# Plot Channel 1 (EMG) with labeled regions
plt.subplot(3, 1, 1)
plt.plot(emg_data.index, emg_data.iloc[:, 0], label="Channel 1", color='blue', linewidth=1)
plt.title("Channel 1 (EMG)")

# Highlight the movement regions on Channel 1
for idx, (start, end) in enumerate(merged_noise_events):
    color = 'red' if idx % 2 == 0 else 'green'  # Alternate colors for movement 1 and movement 2
    plt.axvspan(start, end, color=color, alpha=0.3)

# Highlight Rest regions
plt.axvspan(last_movement_end, len(audio_trigger), color='gray', alpha=0.3, label='Rest')

# Plot Channel 2 (EMG) with labeled regions
plt.subplot(3, 1, 2)
plt.plot(emg_data.index, emg_data.iloc[:, 1], label="Channel 2", color='orange', linewidth=1)
plt.title("Channel 2 (EMG)")

# Highlight the movement regions on Channel 2
for idx, (start, end) in enumerate(merged_noise_events):
    color = 'red' if idx % 2 == 0 else 'green'  # Alternate colors for movement 1 and movement 2
    plt.axvspan(start, end, color=color, alpha=0.3)

# Highlight Rest regions
plt.axvspan(last_movement_end, len(audio_trigger), color='gray', alpha=0.3, label='Rest')

# Plot the Audio Trigger signal on a separate subplot
plt.subplot(3, 1, 3)
plt.plot(audio_trigger.index, audio_trigger, label="Audio Trigger", color='black', linewidth=1)
plt.title("Audio Trigger")

# Highlight noise events in the audio signal
for start, end in merged_noise_events:
    plt.axvspan(start, end, color='yellow', alpha=0.3, label="Noise Event")

# Add a legend to indicate what each color represents
plt.subplot(3, 1, 1)
plt.legend(["Channel 1", "Movement 1", "Movement 2", "Rest"], loc="upper right")
plt.subplot(3, 1, 2)
plt.legend(["Channel 2", "Movement 1", "Movement 2", "Rest"], loc="upper right")
plt.subplot(3, 1, 3)
plt.legend(["Audio Trigger", "Noise Event"], loc="upper right")

# Show the plot
plt.tight_layout()
plt.show()
