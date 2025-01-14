import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_file = 'music files\guitarLoop.mp3'
y, sr = librosa.load(audio_file)

# Define a range of time for zooming into a specific section (for example, 30 to 60 seconds)
start_time = 30  # 30 seconds
end_time = 60  # 60 seconds
start_sample = librosa.time_to_samples(start_time)
end_sample = librosa.time_to_samples(end_time)

# Crop the audio signal to focus on the segment with high activity
y_segment = y[start_sample:end_sample]

# Create the spectrogram for the segment
D_segment = librosa.amplitude_to_db(librosa.stft(y_segment), ref=np.max)

# Create the waveform for the segment (for comparison)
time_segment = np.linspace(start_time, end_time, len(y_segment))

# Plot the spectrogram and overlay the waveform
plt.figure(figsize=(12, 8))

# Spectrogram
plt.subplot(2, 1, 1)
librosa.display.specshow(D_segment, x_axis='time', y_axis='log', sr=sr, x_coords=np.linspace(start_time, end_time, D_segment.shape[-1]))
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (Zoomed into 30s - 60s of Pirates of the Caribbean)')

# Waveform overlay
plt.subplot(2, 1, 2)
plt.plot(time_segment, y_segment, color='r', alpha=0.5)
plt.title('Waveform (Zoomed into 30s - 60s of Pirates of the Caribbean)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

# Show the plots
plt.tight_layout()
plt.show()
