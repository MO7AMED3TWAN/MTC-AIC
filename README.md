# Audio Processing Project

## Overview

This project provides a comprehensive framework for audio processing tasks, including noise reduction and silence removal from audio files. It utilizes Python libraries such as `librosa`, `noisereduce`, `pydub`, and `matplotlib` for signal processing, noise reduction, audio visualization, and file handling.

## Structure

The project is organized into several modules:

- **`audio_processor.py`**: Contains the `AudioProcessor` class, which provides functions for reading audio files, detecting noise, and removing silence using the `pydub` library.
  
- **`plotter.py`**: Defines the `Plotter` class for generating waveform and spectrogram visualizations using `matplotlib` and `librosa`.

- **`wave_reader.py`**: Implements the `WaveReader` class, which orchestrates the processing pipeline. It reads data from CSV files, applies noise reduction and silence removal using methods from `audio_processor.py`, and utilizes `plotter.py` for visualization.

- **`utils.py`**: Includes utility functions for loading data from CSV files and creating output directories.

- **`main.py`**: The main script where the project workflow is orchestrated. It loads data, instantiates necessary classes (`AudioProcessor`, `Plotter`, `WaveReader`), and processes audio files by calling appropriate methods.

## Usage

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/audio-processing-project.git
   cd audio-processing-project
   ```
2. Install dependencies:
  pip install -r requirements.txt

### Running the Project
    
- Update train.csv and adapt.csv with your audio file information.
  
- Place audio files in train/ and adapt/ directories.

## Run main.py to execute the processing pipeline:
   ```bash
    python main.py
  ```
- Processed audio files will be saved in train_noise_reduced/ and adapt_noise_reduced/ directories.

### Components

    AudioProcessor: Handles audio file operations, noise detection, and silence removal.

    Plotter: Provides functions for visualizing audio waveforms and spectrograms.

    WaveReader: Orchestrates the processing pipeline, reading data, applying noise reduction, removing silence, and generating visualizations.

### Dependencies

    librosa: Audio processing library for loading, trimming, and visualizing audio files.
    noisereduce: Library for noise reduction using spectral gating.
    pydub: Manipulation of audio files, including silence detection and removal.
    matplotlib: Plotting library for visualizing waveforms and spectrograms.

### Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Mention any individuals or organizations that have contributed to the project or inspired your work.
