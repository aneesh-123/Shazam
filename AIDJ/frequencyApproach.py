import numpy as np
import scipy.signal as signal
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'music files\juiceWrldRepeat.mp3'
audio, sr = librosa.load(audio_path, sr=None)  # sr=None keeps the original sampling rate

# Define frequency ranges (in Hz)
low_freq = 200   # Low frequencies (Bass)
high_freq = 5000 # High frequencies (Treble)

# Improved Low-pass filter (for low frequencies)
def low_pass_filter(audio, cutoff, sr, order=6):
    nyquist = 0.5 * sr  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)

# Improved High-pass filter (for high frequencies)
def high_pass_filter(audio, cutoff, sr, order=6):
    nyquist = 0.5 * sr  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)

# Improved Band-pass filter (for mid frequencies)
def band_pass_filter(audio, low_cutoff, high_cutoff, sr, order=6):
    nyquist = 0.5 * sr
    normal_low_cutoff = low_cutoff / nyquist
    normal_high_cutoff = high_cutoff / nyquist
    b, a = signal.butter(order, [normal_low_cutoff, normal_high_cutoff], btype='band')
    return signal.filtfilt(b, a, audio)

# Apply the filters to separate frequency bands
low_freq_audio = low_pass_filter(audio, low_freq, sr)
high_freq_audio = high_pass_filter(audio, high_freq, sr)
mid_freq_audio = band_pass_filter(audio, low_freq, high_freq, sr)

# Compute the spectrogram for a given audio signal
def compute_spectrogram(audio, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    return D

# Compute the spectrograms for the original and each frequency band
original_spectrogram = compute_spectrogram(audio, sr)
low_spectrogram = compute_spectrogram(low_freq_audio, sr)
mid_spectrogram = compute_spectrogram(mid_freq_audio, sr)
high_spectrogram = compute_spectrogram(high_freq_audio, sr)

# Plotting the spectrograms for each frequency band and the original signal
plt.figure(figsize=(12, 10))

# Plot original spectrogram
plt.subplot(4, 1, 1)
librosa.display.specshow(original_spectrogram, x_axis='time', y_axis='log', sr=sr, cmap='viridis')
plt.title('Original Audio Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')

# Plot low-frequency spectrogram
plt.subplot(4, 1, 2)
librosa.display.specshow(low_spectrogram, x_axis='time', y_axis='log', sr=sr, cmap='Blues')
plt.title('Low Frequency Component Spectrogram (0 - 200 Hz)')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')

# Plot mid-frequency spectrogram
plt.subplot(4, 1, 3)
librosa.display.specshow(mid_spectrogram, x_axis='time', y_axis='log', sr=sr, cmap='Greens')
plt.title('Mid Frequency Component Spectrogram (200 Hz - 5000 Hz)')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')

# Plot high-frequency spectrogram
plt.subplot(4, 1, 4)
librosa.display.specshow(high_spectrogram, x_axis='time', y_axis='log', sr=sr, cmap='Reds')
plt.title('High Frequency Component Spectrogram (5000 Hz and above)')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()
