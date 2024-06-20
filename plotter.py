import matplotlib.pyplot as plt
import librosa.display

class Plotter:
    @staticmethod
    def plot_waveform(audio, sr, title="Waveform"):
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    @staticmethod
    def plot_spectrogram(audio, sr, title="Spectrogram"):
        plt.figure(figsize=(10, 4))
        spectrogram = librosa.stft(audio)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()
