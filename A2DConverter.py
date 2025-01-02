import numpy as np
import librosa
import matplotlib.pyplot as plt

# Function to perform quantization
def quantize(signal, levels):
    # Normalize the signal to the range [0, 1]
    normalized_signal = (signal + 1) / 2  # Shift from [-1, 1] to [0, 1]
    
    # Scale the normalized signal to the range [0, levels-1]
    quantized_signal = np.round(normalized_signal * (levels - 1))  # Round to nearest level
    
    # Map back to the range [0, 1]
    quantized_signal = quantized_signal / (levels - 1)  # Rescale to [0, 1]
    
    # Shift back to the original range [-1, 1]
    quantized_signal = quantized_signal * 2 - 1  # Map back to [-1, 1]
    
    return quantized_signal

# Function to plot the original and quantized waveforms
def plot_waveforms(analog_waveform, sampled_waveform, quantized_waveform, sampling_rate):
    plt.figure(figsize=(10, 8))

    # Plot the original analog waveform (continuous)
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0, len(analog_waveform) / sampling_rate, len(analog_waveform)), analog_waveform)
    plt.title('Original Analog Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the sampled waveform (discrete samples)
    plt.subplot(3, 1, 2)
    plt.stem(np.linspace(0, len(sampled_waveform) / sampling_rate, len(sampled_waveform)), 
             sampled_waveform, basefmt=" ", use_line_collection=True, linefmt='C1-', markerfmt='C1o')
    plt.title('Sampled Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot the quantized waveform
    plt.subplot(3, 1, 3)
    plt.stem(np.linspace(0, len(quantized_waveform) / sampling_rate, len(quantized_waveform)), 
             quantized_waveform, basefmt=" ", use_line_collection=True, linefmt='C2-', markerfmt='C2o')
    plt.title('Quantized Waveform')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# Main function to load the audio file, process it and display waveforms
def process_audio(file_path, sampling_rate=1, quantization_levels=100):
    # Load the audio file using librosa
    audio, sr = librosa.load(file_path, sr=sampling_rate)

    # Create the analog waveform (this is the original audio)
    analog_waveform = audio

    # Sample the audio at the desired sampling rate
    # (In this case, librosa already provides a downsampled version of the audio)
    sampled_waveform = audio

    # Quantize the sampled waveform
    quantized_waveform = quantize(sampled_waveform, quantization_levels)

    # Plot the analog, sampled, and quantized waveforms
    plot_waveforms(analog_waveform, sampled_waveform, quantized_waveform, sr)

# Example of usage
if __name__ == "__main__":
    # Provide the path to your audio file (WAV, MP3, etc.)
    audio_file_path = 'music files/sample1.mp3'  # Replace with your file path
    print("got here")
    process_audio(audio_file_path)
    print("processed")
