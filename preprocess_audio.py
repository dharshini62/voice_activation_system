import librosa
import soundfile as sf
import numpy as np
import os

input_folder = "positive_dataset"
output_folder = "processed_positive"

os.makedirs(output_folder, exist_ok=True)

target_length = 16000

for filename in os.listdir(input_folder):

    if filename.endswith(".wav"):

        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)

        # Load audio as mono at 16 kHz
        audio, sr = librosa.load(
            input_file,
            sr=16000,
            mono=True
        )

        # Make audio exactly 1 second
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        # Normalize audio
        max_value = np.max(np.abs(audio))

        if max_value > 0:
            audio = audio / max_value

        # Save processed audio
        sf.write(
            output_file,
            audio,
            16000,
            subtype="PCM_16"
        )

        print("Processed:", filename)

print("\nAudio preprocessing completed!")
print("Saved in:", output_folder)