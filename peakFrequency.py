import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_file = 'music files\Gmajor.mp3'  # Replace with the path to your audio file
y, sr = librosa.load(audio_file)

# Generate the spectrogram using STFT
D = librosa.stft(y)

# Convert to decibel scale (for better visualization)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Parameters for the windowing strategy
window_duration = 1.0  # Window size in seconds
num_labels_per_window = 3  # Number of peaks to label per window
n_frames = D_db.shape[1]  # Total number of time frames in the spectrogram
frame_rate = librosa.frames_to_time(1, sr=sr)  # Duration of one frame in seconds
frames_per_window = int(window_duration / frame_rate)  # Number of frames per window

# Display the spectrogram
plt.figure(figsize=(12, 8))
librosa.display.specshow(D_db, x_axis='time', y_axis='log', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

# Iterate through each window
for start_frame in range(0, n_frames, frames_per_window):
    # Define the range of frames for this window
    end_frame = min(start_frame + frames_per_window, n_frames)
    
    # Extract the relevant portion of the spectrogram
    D_db_window = D_db[:, start_frame:end_frame]
    
    # Find the peaks in this window
    threshold = np.mean(D_db_window) + 10  # Use a threshold 10 dB above the mean value
    peaks_window = (D_db_window > threshold)  # Boolean array for peaks
    
    # Get indices of the peaks
    peak_freqs, peak_times = np.where(peaks_window)
    
    # Convert indices to time and frequency values
    times = librosa.frames_to_time(peak_times + start_frame, sr=sr)
    frequencies = librosa.fft_frequencies(sr=sr)[peak_freqs]
    
    # Filter to select the most prominent peaks in this window
    magnitudes = D_db_window[peak_freqs, peak_times]
    if len(magnitudes) > 0:
        prominent_indices = np.argsort(magnitudes)[-num_labels_per_window:]  # Top peaks
        
        # Get the prominent times and frequencies
        prominent_times = times[prominent_indices]
        prominent_frequencies = frequencies[prominent_indices]
        
        # Annotate the spectrogram
        used_positions = []  # Track used positions to avoid overlap
        for t, f in zip(prominent_times, prominent_frequencies):
            plt.plot(t, f, 'ro')  # Mark the peak with a red dot

            # Adjust vertical position to avoid overlap
            offset = 0
            for used_t, used_f in used_positions:
                if abs(used_t - t) < 0.05 and abs(used_f - f) < 50:  # Close to an existing label
                    offset += 50  # Shift the label upwards
            
            # Add text annotation
            plt.text(
                t, f + offset, f"{int(f)} Hz", color="white", fontsize=10, 
                fontweight="bold", ha='left', va='bottom'
            )
            used_positions.append((t, f + offset))  # Update used positions
        print(used_positions)
# Show the annotated spectrogram
plt.show()
q