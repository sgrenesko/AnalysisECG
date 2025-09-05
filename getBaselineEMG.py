import csv
import serial
import time

# This function attempts to connect to an Arduino on common serial ports.
# Should work on Windows, Linux, and macOS.
def connect_arduino(baud=115200, timeout=1):
    ports = [
        "COM3",                   # Windows default
        "/dev/ttyUSB0",           # Linux USB
        "/dev/ttyACM0",           # Linux ACM
        "/dev/tty.usbmodem1101",  # macOS USB modem
        "/dev/tty.usbserial-1101" # macOS USB serial
    ]
    
    for port in ports:
        try:
            arduino = serial.Serial(port, baud, timeout=timeout)
            print(f"Connected on {port}")
            return arduino
        except Exception:
            continue

    raise IOError("No device found on common ports.")

arduino = connect_arduino() # Device successfully connected, now data passed into arduino object
