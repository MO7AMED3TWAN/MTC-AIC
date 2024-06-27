from utils import *

# Now, calling the load_data function
train_csv = "./data/train.csv"  # Replace with the actual path to your CSV file
train_audio_dir = "./data/train"  # Replace with the directory containing your audio files

train_dataset, validation_dataset = load_data(train_csv, train_audio_dir)

# Calculate the size of the training dataset
train_size = sum(1 for _ in train_dataset.unbatch())

# Calculate the size of the validation dataset
validation_size = sum(1 for _ in validation_dataset.unbatch())

print(f"Size of the training set: {train_size}")
print(f"Size of the validation set: {validation_size}")

for spectrogram, label in train_dataset.take(1):  # Example of taking one batch
    print("Spectrogram shape:", spectrogram.shape)
    print("Label shape:", label.shape)
