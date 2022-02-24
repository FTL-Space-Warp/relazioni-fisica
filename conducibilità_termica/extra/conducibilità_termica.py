import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


# Prima misura (presa per intero per poter disegnare il grafico)
t1r = np.loadtxt("temperature-WI/misura1.txt", usecols=0)
T1r = np.loadtxt("temperature-WI/misura1.txt", usecols=1)
t1a = np.loadtxt("temperature-WI/misura1.txt", usecols=2)
T1a = np.loadtxt("temperature-WI/misura1.txt", usecols=3)


def temp(x, T0, k):
    return T0 + k*x


# Le temperature finali
xr = np.array([45.5, 41.3, 39.2, 37.1, 34.9, 32.8, 30.6, 28.5, 26.3, 24.2, 22,
               19.8, 17.7, 15.6, 13.4, 11.3, 9.2, 7, 4.8, 2.7])
xa = np.array([38, 35.5, 33, 30.5, 28, 25.5, 23, 20.5, 18, 15.5, 13, 10.5, 8,
               5.5, 3])
sigma_x = 0.01
Tr = []
Ta = []
sigma_T = 0.4
for i in range(20):
    Tr.append(np.loadtxt("temperature-WI/misura"+str(i+1)+".txt",
                         usecols=1)[-1])
    if i <= 14:
        Ta.append(np.loadtxt("temperature-WI/misura"+str(i+1)+".txt",
                             usecols=3)[-1])

# Grafico della prima misura

prima_misura = plt.figure("Prima misura")
plt.plot(t1r, T1r, c="red", label="Rame")
plt.plot(t1a, T1a, c="blue", label="Alluminio")
plt.legend()

plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('t [s]')
plt.ylabel('T [C]')
plt.savefig("prima_misura.pdf")

# Grafico delle temperature

temperature = plt.figure("Temperature")
temperature.add_axes((0.1, 0.3, 0.8, 0.6))
plt.errorbar(xr, Tr, yerr=sigma_T, xerr=sigma_x, fmt=".", c="red")
plt.errorbar(xa, Ta, yerr=sigma_T, xerr=sigma_x, fmt=".", c="blue")

# Fit delle temperature
popt_r, pcov = curve_fit(temp, xr, Tr)
T0r_hat, kr_hat = popt_r
chisq_r = np.sum(((temp(xr, *popt_r) - Tr)/sigma_T)**2)
plt.plot(xr, temp(xr, *popt_r), c="red", label="Rame")
print("# Rame")
print("sigma_T0 = %f\nsigma_k = %f" % tuple(np.sqrt(np.diag(pcov))))
print(f"chisq = {chisq_r}")

popt_a, pcov = curve_fit(temp, xa, Ta)
T0a_hat, ka_hat = popt_a
chisq_a = np.sum(((temp(xa, *popt_a) - Ta)/sigma_T)**2)
plt.plot(xa, temp(xa, *popt_a), c="blue", label="Alluminio")
print("# Alluminio")
print("sigma_T0 = %f\nsigma_k = %f" % tuple(np.sqrt(np.diag(pcov))))
print(f"chisq = {chisq_a}")

plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('x [cm]')
plt.ylabel('T [C]')
plt.legend()

# Grafico dei residui

temperature.add_axes((0.1, 0.1, 0.8, 0.2))

res_r = Tr - temp(xr, *popt_r)
plt.errorbar(xr, res_r, sigma_T, fmt="o", markersize=4, c="red")

res_a = Ta - temp(xa, *popt_a)
plt.errorbar(xa, res_a, sigma_T, fmt="o", markersize=4, c="blue")

plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("x [cm]")
plt.ylabel("Residuai")

plt.savefig("temperature.pdf")


plt.show()
