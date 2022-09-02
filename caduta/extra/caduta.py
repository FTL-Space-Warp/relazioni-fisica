import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = sys.argv[1]

# I dati nel file sono in "screen coordinates"
# risulta che 1m = 1346.39 screen coordinates

h = np.loadtxt(file_path) / 1346.39
h = h.max() - h  # sposto l'origine ed inverto le coordinate
sigma_h = np.full(h.shape, 0.003)

t = np.arange(0, 0.008*len(h), 0.008)


def parabola(t, v0, g):
    """Modello per la caduta del grave
    """
    return (0.5 * g * t ** 2) + v0 * t + h[0]


fig = plt.figure('Altezza')

# Scatter plot dei dati.
plot = fig.add_axes((0.15, 0.3, 0.8, 0.6))
plt.errorbar(t, h, yerr=sigma_h, fmt='.')

# Fit
popt, pcov = curve_fit(parabola, t, h, p0=[-1, -9.81], sigma=sigma_h)
v0_hat, g_hat = popt
sigma_v0, sigma_g = np.sqrt(pcov.diagonal())

print(f"v0 = {v0_hat} ± {sigma_v0}\n\
g = {g_hat} ± {sigma_g}")

chi2 = np.sum(((h - parabola(t, v0_hat, g_hat)) / sigma_h) ** 2)
ndof = len(h) - 2
print(f"Chi2/dof: {chi2}±{(2*ndof)**.5}/{ndof}")

plt.text(0.005, 0.25, fr"$\chi^2 /\nu:{round(chi2, 1)}\pm"
         fr"{round((2*ndof)**.5, 1)}/{ndof}$",
         fontdict={"usetex": True}, fontsize=20)

# Grafico del modello di best-fit.
x = np.linspace(t.min(), t.max(), 100)
plt.plot(x, parabola(x, v0_hat, g_hat))
plt.xlabel('t [s]')
plt.ylabel('h [m]')
plt.grid(which='both', ls='dashed', color='gray')

# Grafico dei residui
fig.add_axes((0.15, 0.1, 0.8, 0.2), xlim=plot.get_xlim())
plt.grid(which="both", ls="dashed", color="gray")
res = h - parabola(t, v0_hat, g_hat)
plt.errorbar(t, res, yerr=sigma_h, fmt="o")
plt.xlabel('t [s]')
plt.ylabel('Residui [m]')


plt.savefig('altezza.pdf')

plt.show()
