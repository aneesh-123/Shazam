import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Helper function to match frequencies to musical notes
def frequency_to_note(freq):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if freq == 0:
        return "N/A"
    note_number = int(round(12 * np.log2(freq / 440.0) + 69))
    note_index = note_number % 12
    octave = (note_number // 12) - 1
    note = note_names[note_index]
    return f"{note}{octave}"

# Load the audio file
audio_file = 'music files\Pirates.mp3'  # Replace with the path to your audio file
y, sr = librosa.load(audio_file, sr=None)

# Short-Time Fourier Transform (STFT)
D = librosa.stft(y)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Get the spectrogram data
spectrogram = np.abs(D)
times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr)  # Time axis
frequencies = librosa.fft_frequencies(sr=sr)  # Frequency axis

# Initialize a list to store the constellation points
constellation_points = []

# Find peaks for each time frame
for t in range(0, spectrogram.shape[1], 70):  # Iterate over time frames
    frame = spectrogram[:, t]
    # Detect peaks in the current frame (frequency axis)
    peak_indices, _ = find_peaks(frame, height=np.max(frame) * 0.2)  # Adjust height threshold as needed
    sample = peak_indices[:4]
    for peak in sample:
        constellation_points.append((times[t], frequencies[peak]))  # Store time and frequency

# Convert to NumPy array for easier handling
constellation_points = np.array(constellation_points)

# Plot the constellation map
plt.figure(figsize=(12, 8))
plt.scatter(constellation_points[:, 0], constellation_points[:, 1], color='white', s=10, label='Peaks')
plt.title('Constellation Map', fontsize=20)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Frequency (Hz)', fontsize=16)
plt.yscale('log')  # Use a logarithmic scale for frequency
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend()
plt.gca().set_facecolor('black')  # Set background to black for better visibility
plt.colorbar(label="Magnitude (dB)")
plt.show()

# Print the identified constellation points (optional)
print("Constellation Points (Time, Frequency):")
for time, freq in constellation_points:
    note = frequency_to_note(freq)
    print(f"Time: {time:.2f}s, Frequency: {freq:.2f} Hz -> Note: {note}")
