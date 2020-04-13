import librosa
import numpy as np
import sounddevice
from matplotlib import colors


class BeatDetector:
    def __init__(self, sample_rate=44100, length=8):
        """
        The BeatDetector as the name claims, handles the beat detection.

        Args:
            sample_rate: Samples per second. This is the standard samplerate, only change this if you really know what
            you are doing.
            length: Length of a sample in seconds.
        """
        self.samplerate = sample_rate
        self.blocksize = sample_rate * length
        self.stream = sounddevice.InputStream(
            channels=2, samplerate=sample_rate, blocksize=self.blocksize, clip_off=True
        )
        self.stream.start()

    def detect_beat(self, **kwargs):
        """
        Detects the current BPM if there are enough samples currently in the stream. A longer blocksize results in a
        more accurate estimation.

        Args:
            **kwargs: Beat tempo options, like previous bpm can be passed here to librosa.

        Returns:
            Estimate of the BPM.

        """
        if self.has_enough_samples():
            while self.has_enough_samples():
                data, overflowed = self.stream.read(self.blocksize)
            sound_sample = np.mean(data, axis=1)
            onset_env = librosa.onset.onset_strength(sound_sample, sr=self.samplerate)
            bpm = librosa.beat.tempo(
                sound_sample, sr=self.samplerate, onset_envelope=onset_env, **kwargs
            )[0]
            return bpm
        else:
            return -1

    def has_enough_samples(self):
        return self.stream.read_available > self.blocksize


def get_color(col_name: str):
    color = colors.to_rgba(col_name)
    return color
