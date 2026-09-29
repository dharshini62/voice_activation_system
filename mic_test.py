import sounddevice as sd

print("Recording started... Speak now!")

audio = sd.rec(16000, samplerate=16000, channels=1)
sd.wait()

print("Recording completed!")