import sys
import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

audio_path = sys.argv[1]
ignore = int(sys.argv[2]) if len(sys.argv) > 2 else 0
csv_path = audio_path.rstrip('wav') + 'csv'
stream = wave.open(audio_path)
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
framerate = stream.getframerate()

t = np.loadtxt(csv_path)[ignore:]

plt.figure('Rimbalzi pallina')
plt.plot(np.arange(len(signal))/framerate, signal)
plt.xlabel('Tempo [s]')
plt.ylabel('Ampiezza [dB]')
plt.savefig('audio_rimbalzi.pdf')


sigma_t = 0.002

dt = np.diff(t)

n = np.arange(len(dt)) + 1 + ignore

h = 9.81 * (dt**2.) / 8.0
dh = 2.0 * np.sqrt(2.0) * h * sigma_t / dt


def expo(n, h0, gamma):
    return h0 * (gamma ** n)


fig = plt.figure('Altezza dei rimbalzi')
if ignore:
    fig.add_axes((0.15, 0.3, 0.8, 0.6), xlim=(ignore, 50),
                 ylim=(0.0007, .16), xticklabels=[])
else:
    fig.add_axes((0.15, 0.3, 0.8, 0.6), xlim=(ignore, 50), xticklabels=[])

plt.errorbar(n, h, dh, fmt='o')
popt, pcov = curve_fit(expo, n-ignore, h, sigma=dh, p0=(1, .5))
h0_hat, gamma_hat = popt
sigma_h0, sigma_gamma = np.sqrt(pcov.diagonal())
print("h0_hat: %f" % h0_hat)
print("sigma_h0: %f" % sigma_h0)
print("gamma_hat: %f" % gamma_hat)
print("sigma_gamma: %f" % sigma_gamma)

x = np.linspace(1.0, float(len(n)), 1000) + ignore
plt.plot(x, expo(x-ignore, h0_hat, gamma_hat))
plt.yscale('log')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Rimbalzo')
plt.ylabel('Altezza massima [m]')

if ignore:
    fig.add_axes((0.15, 0.1, 0.8, 0.2), xlim=(ignore, 50),
                 ylim=(-0.01, 0.01))
else:
    fig.add_axes((0.15, 0.1, 0.8, 0.2), xlim=(ignore, 50), ylim=(-0.1, 0.1))

res = h - expo(n-ignore, h0_hat, gamma_hat)
plt.errorbar(n, res, dh, fmt="o", markersize=4)
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Rimbalzo")
plt.ylabel("Residui")

if ignore:
    plt.savefig('altezza_rimbalzi[%d:].pdf' % ignore)
else:
    plt.savefig('altezza_rimbalzi.pdf')

plt.show()
