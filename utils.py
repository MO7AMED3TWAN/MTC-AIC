import pandas as pd
import os

def load_data(train_csv, adapt_csv, train_audio_dir, adapt_audio_dir):
    train_df = pd.read_csv(train_csv)
    adapt_df = pd.read_csv(adapt_csv)

    train_df['audio_path'] = train_df['audio'].apply(lambda x: os.path.join(train_audio_dir, x + '.wav'))
    adapt_df['audio_path'] = adapt_df['audio'].apply(lambda x: os.path.join(adapt_audio_dir, x + '.wav'))

    return train_df, adapt_df

def create_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
