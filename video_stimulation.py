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

if __name__ == "__main__":
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)  # Give the Teensy time to reset after serial opens

    try:
        set_pwm(1)
        time.sleep(3)
        set_pwm(0)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        ser.close()
        print("Serial connection closed.")
