import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = sys.argv[1]

length = 1.05

d = np.loadtxt(file_path, delimiter=',', usecols=0,
               converters={0: lambda d: abs(float(d)/100-length/2)})
sigma_d = np.full(np.shape(d), 0.01)

T = np.mean(np.loadtxt(file_path, delimiter=',', usecols=(1, 2, 3, 4, 5))/10,
            axis=1)
sigma_T = np.full(np.shape(T), 0.02)


def period_model(d, L, g=9.81):
    """Modello per il periodo del pendolo.
    """
    return 2.0 * np.pi * np.sqrt(((L**2.0) / 12.0 + d**2.0) / (g * d))


fig = plt.figure('Periodo')

# Scatter plot dei dati.
graph = fig.add_axes((0.1, 0.3, 0.8, 0.6), xticklabels=[])
plt.errorbar(d, T, yerr=sigma_T, fmt='.')

# Fit
popt, pcov = curve_fit(period_model, d, T, sigma=sigma_T)
l_hat = popt[0]
sigma_l = np.sqrt(pcov[0, 0])

print(f"l = {l_hat} ± {sigma_l}")

# Grafico del modello di best-fit.
x = np.linspace(0.02, 0.5, 100)
plt.plot(x, period_model(x, l_hat))
plt.xlabel('d [m]')
plt.ylabel('Periodo [s]')
plt.grid(which='both', ls='dashed', color='gray')

chi2 = np.sum(((T - period_model(d, l_hat)) / sigma_T) ** 2)
ndof = len(T) - 1
print(f"Chi2/dof: {chi2}±{(2*ndof)**.5}/{ndof}")

plt.text(0.22, 2.6, fr"$\chi^2 /\nu:{round(chi2, 1)}\pm"
         fr"{round((2*ndof)**.5, 1)}/{ndof}$",
         fontdict={"usetex": True}, fontsize=20)

# Grafico dei residui
fig.add_axes((0.1, 0.1, 0.8, 0.2), xlim=graph.get_xlim(), ylim=(-0.1, 0.1))
plt.grid(which="both", ls="dashed", color="gray")
res = T - period_model(d, l_hat)
plt.errorbar(d, res, sigma_T, fmt="o", markersize=4)
plt.xlabel('d [m]')
plt.ylabel('Residui')

plt.savefig('Periodo_lunghezza.pdf')

plt.show()
