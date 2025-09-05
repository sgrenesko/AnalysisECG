/*
    AnalysisEMG-EMGReader v1.0
    This program reads EMG data from an EMG sensor connected to an Arduino microcontroller.
    It constantly collects the data and outputs it via UART serial communication for
    further analysis to a connected computer.
*/

const byte EMG_READ = A0;
float emgData = 0.0;


void setup(){
    Serial.begin(115200);
    pinMode(EMG_READ, INPUT_PULLUP);
}

void loop(){
    emgData = analogRead(EMG_READ);
    Serial.println(emgData);
    delay(10);
}
