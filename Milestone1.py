
# Sinusoidal Wave
import numpy as np
import matplotlib.pyplot as plt

# Generate a continuous sinusoidal wave
t = np.linspace(0, 1, 1000)  # Time axis
frequency = 5  # Frequency in Hz
amplitude = 1  # Amplitude
sin_wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Plot the sinusoidal wave
plt.figure(figsize=(10, 5))
plt.plot(t, sin_wave, label='Sinusoidal Wave')
plt.title("Sinusoidal Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

# Square Wave
from scipy.signal import square

# Generate a continuous square wave
square_wave = amplitude * square(2 * np.pi * frequency * t)

# Plot the square wave
plt.figure(figsize=(10, 5))
plt.plot(t, square_wave, label='Square Wave', color='orange')
plt.title("Square Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

# Sampling the sinusoidal wave
sampling_rate = 50  # Samples per second
sampled_t = np.linspace(0, 1, sampling_rate)
sampled_wave = amplitude * np.sin(2 * np.pi * frequency * sampled_t)

# Plot original and sampled signals
plt.figure(figsize=(10, 5))
plt.plot(t, sin_wave, label='Original Signal', alpha=0.7)
plt.stem(sampled_t, sampled_wave, linefmt='r', markerfmt='ro', basefmt=" ", label='Sampled Signal')
plt.title("Sampling")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

# Quantize the sampled signal to 4 levels
quantization_levels = 4
min_val = np.min(sampled_wave)
max_val = np.max(sampled_wave)
quantized_wave = np.round((sampled_wave - min_val) / (max_val - min_val) * (quantization_levels - 1))
quantized_wave = quantized_wave / (quantization_levels - 1) * (max_val - min_val) + min_val

# Plot sampled and quantized signals
plt.figure(figsize=(10, 5))
plt.stem(sampled_t, sampled_wave, linefmt='g', markerfmt='go', basefmt=" ", label='Sampled Signal')
plt.step(sampled_t, quantized_wave, label='Quantized Signal', where='mid', color='red')
plt.title("Quantization")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

from scipy.io.wavfile import read

def plot_wav_file(filename):
    # Load the WAV file
    sampling_rate, data = read(filename)

    # If the data has more than one channel (stereo), select one channel
    if len(data.shape) > 1:
        data = data[:, 0]  # Use the first channel

    # Create a time axis
    duration = len(data) / sampling_rate  # Duration in seconds
    time = np.linspace(0., duration, len(data))

    # Plot the signal
    plt.figure(figsize=(10, 5))
    plt.plot(time, data, label="Audio Signal")
    plt.title(f"Waveform of {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

# Example usage
plot_wav_file(r"C:\Users\ibrah\Documents\synthetic_audio_signal.wav")  # Replace with your WAV file path

# Define the parameters of the equation
A = 1  # Amplitude of sine wave
B = 0.5  # Amplitude of cosine wave
f1 = 5  # Frequency of sine wave (Hz)
f2 = 10  # Frequency of cosine wave (Hz)
duration = 1  # Duration of the signal in seconds
sampling_rate = 1000  # Sampling rate (samples per second)

# Generate the time axis
t = np.linspace(0, duration, int(sampling_rate * duration))

# Define the custom equation
custom_signal = A * np.sin(2 * np.pi * f1 * t) + B * np.cos(2 * np.pi * f2 * t)

# Plot the signal
plt.figure(figsize=(10, 5))
plt.plot(t, custom_signal, label=rf"$y(t) = {A}\sin(2\pi {f1}t) + {B}\cos(2\pi {f2}t)$")
plt.title("Custom Signal Based on Mathematical Equation")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()
