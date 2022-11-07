import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True

file_path = sys.argv[1]

t = np.loadtxt(file_path, usecols=0)
T = np.loadtxt(file_path, usecols=1)
transit_time = np.loadtxt(file_path, usecols=2)

L = 1.110  # m
sigma_L = 0.001  # m
d = 1.137  # m
sigma_d = 0.001  # m
w = 0.01935  # m
sigma_w = 0.00005  # m


v = (w*L)/(transit_time*d)
sigma_v = np.sqrt((L*sigma_w)**2 + (w*sigma_L)**2 + ((w*L*sigma_d)/d)**2) \
    / (transit_time*d)

ampl = np.arccos(1-(v**2)/(2*9.8*L))
sigma_ampl = sigma_v / (2 * v)

# propagazione dell'errore per T nel caso della formula period_model
sigma_T = (np.pi / np.sqrt(9.81)) * np.sqrt((sigma_L*(1+(ampl**2)/16)**2)/L +
                                            (L*(ampl*sigma_ampl**2))/16)


def period_model(theta, L=1.11, g=9.81):
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
plt.errorbar(t, v, yerr=sigma_v, label='Misure sperimentali')

plt.yscale('log')
plt.xlabel("Tempo [s]")
plt.ylabel("Velocità [m/s]")

# Fit
popt, pcov = curve_fit(exp_model, t, v, sigma=sigma_v, p0=0.9)
eta_hat = popt[0]
sigma_eta = np.sqrt(pcov[0, 0])
chisq = sum(((v-exp_model(t, eta_hat))/sigma_v)**2)
ndof = len(t) - 1

print(f"lambda = {eta_hat} ± {sigma_eta}")
print(f"chisq = {chisq}±{(2*ndof)**0.5}/{ndof}")

plt.text(10, 0.15, fr"$\chi^2 /\nu:{round(chisq, 1)}\pm"
         fr"{round((2*ndof)**.5, 1)}/{ndof}$",
         fontdict={"usetex": True}, fontsize=14)

# Grafico del modello di best-fit.
x = np.linspace(0, t[-1], 20)
plt.plot(x, exp_model(x, eta_hat), label='Best fit')
plt.grid(which='both', ls='dashed', color='gray')
plt.legend()

plt.savefig('pendolo_quadrifilare_velocità.pdf')

# Grafico dei periodi
periodo = plt.figure('Periodo')
periodo.add_axes((0.15, 0.1, 0.8, 0.8))
plt.plot(ampl, T, label='Misure sperimentali')

plt.xlabel("Ampiezza [rad]")
plt.ylabel("Periodo [s]")


# Grafico del modello matematico
plt.plot(ampl, period_model(ampl, L), color='orange', label=r'$T(\theta)$')
plt.plot(ampl, period_model(ampl, L)+sigma_T, color='orange', linestyle='--',
         label=r'$T(\theta)\pm \sigma_{T}$')
plt.plot(ampl, period_model(ampl, L)-sigma_T, color='orange', linestyle='--')
plt.grid(which='both', ls='dashed', color='gray')
plt.legend()


plt.savefig('pendolo_quadrifilare_periodo.pdf')

print(f"v0 = {v[0]}±{sigma_v[0]}")
print(f"sigma_t.max() = {sigma_T.max()}")
plt.show()
