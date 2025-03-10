import joblib
import tsfel
import pandas as pd
import numpy as np
import glob

# Load the trained Random Forest model
rf_model = joblib.load("rf_model.pkl")

# Get all buffer files
buffer_files = glob.glob("buffer_class_*.csv")

# Define TSFEL feature extraction configuration
cfg = tsfel.get_features_by_domain()

# Expected feature names from training (loaded from Jupyter Notebook)
expected_feature_names = rf_model.feature_names_in_  # Extract names used in training

# Test each buffer file
for file in buffer_files:
    # Read the buffer file
    buffer_df = pd.read_csv(file)

    # Extract the expected class from the filename
    expected_class = int(file.split("_")[-1].split(".")[0])

    # Rename columns to match training format (channel_0, channel_1 instead of channel1, channel2)
    buffer_df.columns = [f"channel_{i}" for i in range(buffer_df.shape[1])]

    # Extract TSFEL features
    extracted_features = tsfel.time_series_features_extractor(cfg, buffer_df, fs=1000, verbose=0)

    # Rename extracted feature columns to match training format
    extracted_features.columns = [name.replace("channel1", "channel_0").replace("channel2", "channel_1")
                                  for name in extracted_features.columns]

    # Ensure the feature order matches training
    extracted_features = extracted_features.reindex(columns=expected_feature_names, fill_value=0)

    # Perform classification
    prediction = rf_model.predict(extracted_features)

    # Display results
    print(f"File: {file} | Predicted: {prediction[0]} | Expected: {expected_class}")

print("\n✅ Model testing on buffer files complete.")
