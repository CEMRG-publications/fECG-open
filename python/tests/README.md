# Test Recording

The four binary files in this directory (`test.ch1`–`test.ch4`) are a
2-minute excerpt (segments 0–1, t=0–120 s) of a full abdominal ECG
recording obtained from a healthy volunteer. The full recording is
provided in `./data/inputs/`.

The excerpt was created specifically for automated testing. It is long
enough to exercise two complete 60-second pipeline segments while keeping
test runtime short.

The files are in the native `.ch` binary format read by `MyReadDataq_32`.
Do not modify them — the integration test relies on their exact content.
