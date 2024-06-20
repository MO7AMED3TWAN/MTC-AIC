import os
from IPython.display import Audio, display
import noisereduce as nr

class WaveReader:
    def __init__(self, dataframe, audio_processor, plotter):
        """
        Initialize WaveReader with dataframe, audio processor, and plotter.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing audio file paths and transcripts.
        - audio_processor (AudioProcessor): Instance of AudioProcessor class for audio processing.
        - plotter: Object with methods for plotting audio waveforms and spectrograms.
        """
        self.data = dataframe
        self.audio_processor = audio_processor
        self.plotter = plotter

    def process_random_sample(self, n=1):
        """
        Process a random sample of n audio files from the dataframe.
        Perform noise reduction, plot waveforms and spectrograms, and detect noise.

        Parameters:
        - n (int): Number of random samples to process.
        """
        sample_df = self.data.sample(n)
        for index, row in sample_df.iterrows():
            audio_path = row['audio_path']
            transcript = row['transcript']
            print(f"Processing file: {audio_path}")
            print(f"Transcript: {transcript}")
            
            # Read audio
            audio, sr = self.audio_processor.read_audio(audio_path)
            
            # Plot waveform and spectrogram before noise reduction
            self.plotter.plot_waveform(audio, sr, title=f"Waveform (Before Noise Reduction) - {audio_path}")
            self.plotter.plot_spectrogram(audio, sr, title=f"Spectrogram (Before Noise Reduction) - {audio_path}")
            
            display(Audio(data=audio, rate=sr))
            
            # Perform noise reduction
            reduced_noise = self.audio_processor.reduce_noise(audio, sr)
            
            # Plot waveform and spectrogram after noise reduction
            self.plotter.plot_waveform(reduced_noise, sr, title=f"Waveform (After Noise Reduction) - {audio_path}")
            self.plotter.plot_spectrogram(reduced_noise, sr, title=f"Spectrogram (After Noise Reduction) - {audio_path}")
            
            display(Audio(data=reduced_noise, rate=sr))
            
            # Detect noise after reduction
            noise_detected = self.audio_processor.detect_noise(reduced_noise, sr)
            print(f"Noise detected after reduction: {'Yes' if noise_detected else 'No'}")

    def process_all_files_remove_silence(self, output_dir):
        """
        Process all audio files in the dataframe to remove silence and save to output directory.

        Parameters:
        - output_dir (str): Directory path where the processed audio files will be saved.
        """
        for index, row in self.data.iterrows():
            audio_path = row['audio_path']
            output_path = os.path.join(output_dir, os.path.basename(audio_path))
            print(f"Processing file: {audio_path}")
            self.audio_processor.remove_silence(audio_path, output_path)
