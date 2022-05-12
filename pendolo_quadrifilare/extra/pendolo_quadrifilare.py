import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = sys.argv[1]

t = np.loadtxt(file_path, usecols=0)
T = np.loadtxt(file_path, usecols=1)
transit_time = np.loadtxt(file_path, usecols=2)


v = (0.02*1.077)/(transit_time*1.137)
sigma_v = np.sqrt(((1.077*0.00005)**2 + (1.077*0.02*0.001)**2 +
                   (0.02*0.001)**2) / ((transit_time*1.137)**2))
print(sigma_v)
ampl = np.arccos(1-(v**2)/(2*9.8*1.0707))

sigma_T = np.full(np.shape(T), 0.01)
sigma_ampl = 0.005


def period_model(theta, L, g=9.81):
    """Modello per il periodo del pendolo.
    """
    return 2.0 * np.pi * np.sqrt(L / g) * (1 + (theta ** 2) / 16)


def exp_model(t, eta):
    """Modello per lo smorzamento del pendolo
    """
    return v[0]*(np.exp(-eta*t))


# Grafico delle velocità
velocità = plt.figure('Velocità')
velocità.add_axes((0.1, 0.1, 0.8, 0.8))
plt.errorbar(t, v, yerr=sigma_v)

plt.xlabel("Tempo [s]")
plt.ylabel("Velocità [m/s]")

# Fit
popt, pcov = curve_fit(exp_model, t, v, sigma=sigma_v, p0=0.9)
eta_hat = popt[0]
sigma_eta = np.sqrt(pcov[0, 0])
chisq = sum(((v-exp_model(t, eta_hat))/sigma_v)**2)

print(eta_hat, sigma_eta)
print(chisq)

# Grafico del modello di best-fit.
x = np.linspace(0, t[-1], 20)
plt.plot(x, exp_model(x, eta_hat))
plt.grid(which='both', ls='dashed', color='gray')

plt.savefig('pendolo_quadrifilare_velocità.pdf')

# Grafico dei periodi
periodo = plt.figure('Periodo')
periodo.add_axes((0.15, 0.1, 0.8, 0.8))
plt.plot(ampl, T)

plt.xlabel("Ampiezza [rad]")
plt.ylabel("Periodo [s]")

# Fit
data_slice = 450
popt, pcov = curve_fit(period_model, ampl[:-data_slice], T[:-data_slice],
                       sigma=sigma_T[data_slice:], p0=1)
L_hat = popt[0]
sigma_L = np.sqrt(pcov[0, 0])

print(L_hat, sigma_L)

# Grafico del modello di best-fit.
x = np.linspace(ampl[0], ampl[-data_slice], 20)
plt.plot(x, period_model(x, L_hat))
plt.grid(which='both', ls='dashed', color='gray')


plt.savefig('pendolo_quadrifilare_periodo.pdf')

plt.show()
