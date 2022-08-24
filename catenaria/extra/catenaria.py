import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def catenary(x, a, c, x0):
    return c + a * np.cosh((x - x0) / a)


file_path = "dataset_catenaria.csv"
x, y = np.loadtxt(file_path, delimiter=';',
comments='#', unpack=True)

sigma_y = 0.12  # cm

fig = plt.figure("Fit e residui")
fig.add_axes((0.1, 0.3, 0.8, 0.6))
plt.errorbar(x, y, sigma_y, fmt="o", markersize=4)

popt, pcov = curve_fit(catenary, x, y)
a_hat, c_hat, x0_hat = popt
sigma_a, sigma_c, sigma_x0 = np.sqrt(pcov.diagonal())
print(a_hat, sigma_a, c_hat, sigma_c, x0_hat, sigma_x0)
plt.plot(x, catenary(x, a_hat, c_hat, x0_hat))
plt.grid(which="both", ls="dashed", color="gray")
plt.ylabel("y [cm]")

fig.add_axes((0.1, 0.1, 0.8, 0.2), ylim=(-0.75, 0.75))
res = y - catenary(x, a_hat, c_hat, x0_hat)
plt.errorbar(x, res, sigma_y, fmt="o", markersize=4)
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("x [cm]")
plt.ylabel("Residui")
plt.savefig("catenaria.pdf")

plt.show()
