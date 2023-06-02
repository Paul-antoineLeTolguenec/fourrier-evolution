import numpy as np
import matplotlib.pyplot as plt

# Créer un signal d'exemple
Fs = 1000 # fréquence d'échantillonnage
T = 1/Fs  # période d'échantillonnage
t = np.arange(0,1,T) # intervalle de temps

f = 5    # fréquence du signal
signal = np.sin(2*np.pi*f*t) # signal sinusoïdal

# Calculer la Transformée de Fourier en utilisant numpy
fft = np.fft.fft(signal)

# Calculer les fréquences pour chaque terme du résultat de FFT
N = signal.size # nombre d'échantillons
frequencies = np.fft.fftfreq(N, d=T)

# Tracer le signal original et le spectre en fréquence
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Signal original')

plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft))
plt.title('Spectre en fréquence')
plt.xlabel('Fréquence')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
