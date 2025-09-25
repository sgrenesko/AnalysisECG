# AnalysisECG

An open source program using AI/ML algorithms with lightweight hardware to diagnose conditions generally, utilizing ECG data pulled from skin surface electrodes. 

## How it works

This protocol utilizes UART serial communication between a microcontroller board and a single electromyography (EMG) sensor to collect data from muscle movement. This data is pulled from the sensor via the microcontroller in a readable format, and is then passed through USB connection to a computer running the program. The main program proceeds to sort the data into .csv format, creating a baseline healthy EMG dataset, and another EMG dataset with symptomatic and healthy groups. The .csv data is then fed into an AI/ML model trained on healthy and symptomatic datasets that allow it to parse the EMG readings into a general diagnosis using a Random Forest learning model and comparison. As a reminder, AI models do NOT give true diagnosis, but rather a more precise guess as opposed to a Google search, so do not use this program in place of a doctor. This is for educational, research, and hobbyist use only!

### ***Baseline Reading***
The first pass of the EMG sensor is done to get a baseline reading of a healthy muscle group on a subject for comparison.

### ***Symptomatic Reading***
The second pass of the EMG sensor is done on a symptomatic body part to check for an abnormal reading.

### ***Dataset Comparison***
The program then runs the data through a protocol to determine what exact symptoms are being experienced, what the route cause might be, and examples of next steps to midigate symptoms and what specialist may need to be seen. **AGAIN** This is **NOT** an actual diagnosis please please please please see a doctor if you are having actual problems!!!
