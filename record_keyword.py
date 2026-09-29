import sounddevice as sd
import soundfile as sf

sample_rate = 16000
duration = 1

print("Get ready...")
input("Press ENTER and speak 'Activate'...")

print("Recording...")
audio = sd.rec(
    int(sample_rate * duration),
    samplerate=sample_rate,
    channels=1
)

sd.wait()

sf.write("keyword.wav", audio, sample_rate)

print("Done!")
print("Saved as keyword.wav")