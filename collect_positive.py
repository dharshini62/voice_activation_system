import sounddevice as sd
import soundfile as sf
import os

sample_rate = 16000
duration = 1

os.makedirs("positive_dataset", exist_ok=True)

print("====================================")
print("  ACTIVATE - 100 SAMPLE RECORDING")
print("====================================")
print()
print("Each sample: Press ENTER → say 'Activate'")
print("Recording duration: 1 second")
print()

for i in range(1, 101):

    input(f"Sample {i}/100 - ENTER press pannitu 'Activate' sollunga...")

    print("Recording...")

    audio = sd.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=1
    )

    sd.wait()

    filename = f"positive_dataset/activate_{i:03d}.wav"

    sf.write(
        filename,
        audio,
        sample_rate
    )

    print(f"Saved: {filename}")

print("\n====================================")
print("100 positive samples completed!")
print("Saved in: positive_dataset")
print("====================================")