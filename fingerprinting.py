import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy.signal

# Load the audio file
audio_file = 'music files\guitarLoop.mp3'  # replace with the path to your audio file
y, sr = librosa.load(audio_file)

# Generate the spectrogram using STFT
D = librosa.stft(y)

# Convert to decibel scale (for better visualization)
D_db = librosa.amplitude_to_db(D, ref=np.max)

# Set the threshold for peak detection
threshold = np.mean(D_db) + 10  # Use a threshold 10 dB above the mean value

# Find the peaks
peaks = (D_db > threshold)  # Boolean array where True means there's a peak

# For each peak, record the frequency and time
peak_times, peak_freqs = np.where(peaks)
print(peak_times, peak_freqs)
print(len(peak_times), len(peak_freqs), len(D_db))

# Display the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(D_db, x_axis='time', y_axis='log', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
