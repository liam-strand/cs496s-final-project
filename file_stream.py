import numpy as np
import librosa


class FileStream:
    def __init__(self, file_path, blocksize):
        self.data, self.file_sr = librosa.load(file_path, sr=16000, mono=False)
        if self.data.ndim == 1:
            # Convert mono to stereo by duplicating channels
            self.data = np.stack([self.data, self.data], axis=-1)
        else:
            self.data = self.data.T  # (channels, samples) -> (samples, channels)

        self.blocksize = blocksize
        self.position = 0
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def read(self, frames):
        if self.position >= len(self.data):
            self.finished = True
            return np.zeros((frames, self.data.shape[1]), dtype=np.float32), False

        end = self.position + frames
        chunk = self.data[self.position : end]

        # Pad with zeros if we reached the end
        if len(chunk) < frames:
            padding = np.zeros(
                (frames - len(chunk), self.data.shape[1]), dtype=np.float32
            )
            chunk = np.vstack([chunk, padding])
            self.finished = True

        self.position = end
        return chunk, False
