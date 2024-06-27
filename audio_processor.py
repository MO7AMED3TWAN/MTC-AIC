import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

class AudioProcessor:
    def __init__(self, remove_silence=True):
        self.remove_silence_flag = remove_silence

    def read_audio(self, filepath, sr=None):
        """
        Read audio file from filepath using librosa.

        Parameters:
        - filepath (str): Path to the audio file.
        - sr (int or None): Sampling rate to load the audio file (optional).

        Returns:
        - audio (np.ndarray): Loaded audio data.
        - sr (int): Sampling rate of the loaded audio.
        """
        audio, sr = librosa.load(filepath, sr=sr)
        if self.remove_silence_flag:
            audio, _ = librosa.effects.trim(audio)
        return audio, sr

    def reduce_noise(self, audio, sr, prop_decrease=0.77):
        """
        Reduce noise from audio using noisereduce library.

        Parameters:
        - audio (np.ndarray): Audio data.
        - sr (int): Sampling rate of the audio.
        - prop_decrease (float): Proportion by which to decrease noise (optional).

        Returns:
        - reduced_noise (np.ndarray): Noise-reduced audio data.
        """
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
        return reduced_noise

    def detect_noise(self, audio, sr, threshold=0.02):
        """
        Detect noise in audio.

        Parameters:
        - audio (np.ndarray): Audio data.
        - sr (int): Sampling rate of the audio.
        - threshold (float): Threshold ratio to determine noise (optional).

        Returns:
        - bool: True if noise is detected, False otherwise.
        """
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)
        noise = audio - reduced_noise
        noise_energy = np.sum(noise ** 2)
        audio_energy = np.sum(audio ** 2)
        noise_ratio = noise_energy / audio_energy
        return noise_ratio > threshold

    def remove_silence(self, audio_path, output_path):
        """
        Remove silence from audio file using pydub and save to output_path.

        Parameters:
        - audio_path (str): Path to the input audio file.
        - output_path (str): Path where the processed audio file will be saved.
        """
        audio = AudioSegment.from_file(audio_path, format="wav")
        nonsilent_chunks = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        
        nonsilent_audio = AudioSegment.empty()
        for start, end in nonsilent_chunks:
            nonsilent_audio += audio[start:end]
        
        nonsilent_audio.export(output_path, format="wav")
        print(f"Silence removed using pydub. Output saved at {output_path}")