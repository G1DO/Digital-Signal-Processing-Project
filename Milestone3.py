
import numpy as np
import matplotlib.pyplot as plt
from Milestone2 import filtered_sin_wave,filtered_square_wave
# Define sampling rate
sampling_rate = 1000  # Sampling rate in Hz

# Convolve two signals (Filtered Sinusoidal and Filtered Square Waves)
convolved_signal = np.convolve(filtered_sin_wave, filtered_square_wave, mode='same')

# Time axis for the convolved signal
t_convolved = np.linspace(0, len(convolved_signal) / sampling_rate, len(convolved_signal))

# Plot the convolved signal
plt.figure(figsize=(10, 5))
plt.plot(t_convolved, convolved_signal, label="Convolved Signal")
plt.title("Convolution of Filtered Sinusoidal and Square Waves")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()
# Calculate cross-correlation
from Milestone2 import filtered_sin_wave, filtered_square_wave
correlation = np.correlate(filtered_sin_wave, filtered_square_wave, mode='full')

# Time axis for the correlation
lags = np.arange(-len(filtered_sin_wave) + 1, len(filtered_square_wave))

# Plot the correlation
plt.figure(figsize=(10, 5))
plt.plot(lags, correlation, label="Cross-Correlation")
plt.title("Cross-Correlation of Filtered Sinusoidal and Square Waves")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.legend()
plt.show()
