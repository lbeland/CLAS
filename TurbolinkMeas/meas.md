## Run

```bash
g++ -o measurement measurement.cpp
sudo chrt -f 99 ./measurement 
```

## Observations
- TurboLink Website shows Sample Interval of Recorder (Sample rate that is set in Workspace of Recorder Programm)
- TurboLink DataClient Sample rate can only be changed by stopping the Monitoring in Recorder Programm, changing the fs in Website (has to be lower or equal to Recorder Worksapce fs) and starting monitoring again
- DataClient can only receive packets if Recorder is in Monitoring state