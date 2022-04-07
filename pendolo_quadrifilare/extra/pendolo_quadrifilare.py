import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = sys.argv[1]

T = np.loadtxt(file_path, usecols=1)
transit_time = np.loadtxt(file_path, usecols=2)

print(transit_time)

v= (0.02*1.077)/(transit_time*1.137)
ampl = np.arccos(1-(v**2)/(2*9.8*1.0707))

sigma_T = np.full(np.shape(T), 0.01)
sigma_ampl = 0.005

fig = plt.figure('Periodo')


def period_model(theta, L, g=9.81):
    """Modello per il periodo del pendolo.
    """
    return 2.0 * np.pi * np.sqrt(L / g) * (1 + (theta ** 2) / 16)


# Scatter plot dei dati.
fig.add_axes((0.1, 0.1, 0.8, 0.8))
plt.plot(ampl, T)

plt.xlabel("Ampiezza [rad]")
plt.ylabel("Periodo [s]")

# Fit
popt, pcov = curve_fit(period_model, ampl, T, sigma = sigma_T, p0 = 1)
L_hat = popt[0]
sigma_L = np.sqrt(pcov[0, 0])

print(L_hat, sigma_L)

# Grafico del modello di best-fit.
x = np.linspace(0.02, 0.5, 20)
plt.plot(x, period_model(x, L_hat))
plt.xlabel('Ampiezza [rad]')
plt.ylabel('Periodo [s]')
plt.grid(which='both', ls='dashed', color='gray')


plt.savefig('pendolo_quadrifilare.pdf')

plt.show()
