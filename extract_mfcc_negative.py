import librosa
import numpy as np
import os

input_folder = "processed_negative"
output_folder = "mfcc_negative"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.endswith(".wav"):

        input_file = os.path.join(input_folder, filename)

        # Load processed audio
        audio, sr = librosa.load(
            input_file,
            sr=16000,
            mono=True
        )

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=16000,
            n_mfcc=13,
            n_fft=512,
            hop_length=320
        )

        # Save MFCC as NumPy file
        output_file = os.path.join(
            output_folder,
            filename.replace(".wav", ".npy")
        )

        np.save(output_file, mfcc)

        print("MFCC extracted:", filename)
        print("MFCC shape:", mfcc.shape)

print("\nNegative MFCC extraction completed!")
print("Saved in:", output_folder)