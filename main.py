import os
from audio_processor import AudioProcessor
from plotter import Plotter
from wave_reader import WaveReader
import utils

# Load data
train_csv = './data/train.csv'
adapt_csv = './data/adapt.csv'
train_audio_dir = './data/train/'
adapt_audio_dir = './data/adapt/'

train_df, adapt_df = utils.load_data(train_csv, adapt_csv, train_audio_dir, adapt_audio_dir)

# Instantiate the necessary classes
audio_processor = AudioProcessor()
plotter = Plotter()

# Instantiate the WaveReader class with the training data
wave_reader = WaveReader(train_df, audio_processor, plotter)
# Process a random sample of 3 audio files
wave_reader.process_random_sample(n=3)

# Create output directories if they don't exist
output_audio_dir = 'train_noise_reduced/'
output_audio_dir1 = 'adapt_noise_reduced/'

utils.create_output_dir(output_audio_dir)
utils.create_output_dir(output_audio_dir1)

# Instantiate WaveReader for both train and adapt datasets
wave_reader_train = WaveReader(train_df, audio_processor, plotter)
wave_reader_adapt = WaveReader(adapt_df, audio_processor, plotter)

# Process all files to remove silence and save to the output directory
wave_reader_train.process_all_files_remove_silence(output_audio_dir)
wave_reader_adapt.process_all_files_remove_silence(output_audio_dir1)

# Load CSV files again for noise-reduced audio paths
adapt_df1, train_df1 = utils.load_data(adapt_csv, train_csv, output_audio_dir1, output_audio_dir)

# Instantiate WaveReader with the noise-reduced data
wave_reader_noise_reduced = WaveReader(train_df1, audio_processor, plotter)
# Process a random sample of 3 audio files from the noise-reduced data
wave_reader_noise_reduced.process_random_sample(n=3)
