import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.fftpack import fft

# Helper function to match frequencies to musical notes
def frequency_to_note(freq):
    # List of musical note names
    note_names = [
        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
    ]
    if freq == 0:
        return "N/A"
    # Calculate the note number
    note_number = int(round(12 * np.log2(freq / 440.0) + 69))
    note_index = note_number % 12
    octave = (note_number // 12) - 1
    note = note_names[note_index]
    return f"{note}{octave}"

# Load the audio file
audio_file = 'music files/Gmajor.mp3'  # Replace with the path to your audio file
y, sr = librosa.load(audio_file, sr=None)

# Short-Time Fourier Transform (STFT)
D = librosa.stft(y)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Convert the signal to a frequency spectrum
fft_spectrum = np.abs(fft(y))
frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / sr)

# Only keep the positive half of the spectrum
positive_frequencies = frequencies[:len(frequencies) // 2]
positive_fft_spectrum = fft_spectrum[:len(fft_spectrum) // 2]

# Find peak frequencies
peak_indices, _ = find_peaks(positive_fft_spectrum, height=np.max(positive_fft_spectrum) * 0.1)
peak_frequencies = positive_frequencies[peak_indices]
peak_magnitudes = positive_fft_spectrum[peak_indices]

# Sort peaks by magnitude and select the top ones
sorted_indices = np.argsort(peak_magnitudes)[-6:]  # Top 6 peaks
sorted_peak_frequencies = peak_frequencies[sorted_indices]
sorted_peak_magnitudes = peak_magnitudes[sorted_indices]

# Display the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of G Major Chord')

# Annotate the spectrogram with peak frequencies and notes
for freq in sorted_peak_frequencies:
    note = frequency_to_note(freq)  # Get the note for the frequency
    plt.axhline(freq, color='white', linestyle='--', linewidth=6)
    plt.text(0.5, freq, f"{note} ({freq:.1f} Hz)", color="white", fontsize=15,
            fontweight="bold", ha='right', va='bottom', bbox=dict(facecolor='blue', alpha=0.5))

# Show the spectrogram
plt.show()

# Print the identified frequencies and corresponding notes
print("Identified Peak Frequencies and Notes:")
for freq in sorted_peak_frequencies:
    note = frequency_to_note(freq)
    print(f"Frequency: {freq:.2f} Hz -> Note: {note}")
