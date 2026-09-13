# FM radio parser using an RTL-SDR V4 with a QT GUI

## FM TUNER
![alt text](./mdImages/fmTuner.png)

## RF Waterfall diagram
![alt waterfall](./mdImages/rfWaterfall.png)
## Spectrum Analyzer
![alt spectrum](./mdImages/spectrumAnalyzer.png)
## RDS Packet Info
![alt rds](./mdImages/rdsPacketInfo.png)
For further reference on RDS packets refer to: https://en.wikipedia.org/wiki/Radio_Data_System


## Standard FM broadcast max frequency devication
    MAX_DEVIATION_HZ = 75_000
    FULL_SCALE_RAD_PER_SAMPLE = 2 * pi * MAX_DEVIATION / SDR_SAMPLE_RATE
