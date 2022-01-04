import sys
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

audio_path = sys.argv[1]
csv_path = audio_path.rstrip('wav') + 'csv'
stream = wave.open(audio_path)
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)

t = np.loadtxt(csv_path)

# plt.figure('Rimbalzi pallina')
# plt.plot(np.arange(len(signal)), signal)
# plt.xlabel('Tempo [s]')
# plt.savefig('audio_rimbalzi.pdf')


sigma_t = 0.003

dt = np.diff(t)

n = np.arange(len(dt)) + 1.

h = 9.81 * (dt**2.) / 8.0
dh = 2.0 * np.sqrt(2.0) * h * sigma_t / dt


def expo(n, h0, gamma):
    return h0 * (gamma ** n)


plt.figure('Altezza dei rimbalzi')
plt.errorbar(n, h, dh, fmt='o')
popt, pcov = curve_fit(expo, n, h, sigma=dh, p0=(1, .5))
h0_hat, gamma_hat = popt
sigma_h0, sigma_gamma = np.sqrt(pcov.diagonal())
print(h0_hat, sigma_h0, gamma_hat, sigma_gamma)
x = np.linspace(0.0, float(len(n)), 1000)
plt.plot(x, expo(x, h0_hat, gamma_hat))
plt.yscale('log')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Rimbalzo')
plt.ylabel('Altezza massima [m]')
plt.savefig('altezza_rimbalzi.pdf')

plt.show()
