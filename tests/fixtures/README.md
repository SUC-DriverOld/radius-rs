# Generated codec fixture

`tone.m4a` is 0.1 seconds of a generated 440 Hz sine wave, 44.1 kHz stereo,
encoded as AAC at 64 kbit/s. It contains no third-party recording. The fixture
keeps M4A input tests independent of external tools and AAC encoder dependencies.

To regenerate it from the repository root:

```sh
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=0.1" -ac 2 -c:a aac -b:a 64k -map_metadata -1 tests/fixtures/tone.m4a
```
