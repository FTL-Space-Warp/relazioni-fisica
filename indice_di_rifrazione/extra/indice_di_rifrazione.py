import numpy as np
from matplotlib import pyplot as plt

r = 41*2

# Misure
x_i = np.array([8, 14, 20, 25, 30, 37])*2
x_r = np.array([6, 10, 14, 17, 21, 25])*2

sin_i = x_i / r
sin_r = x_r / r

# Errori
sigma_sin_i = ((0.577)*np.sqrt((x_i**2)+(r**2)))/(x_i*r)
sigma_sin_r = ((0.577)*np.sqrt((x_r**2)+(r**2)))/(x_r*r)
var_n = (sigma_sin_i/sin_r)**2 + (((sin_i*sigma_sin_r)/(sin_r**2)))**2

n = sin_i/sin_r


# Funzioni
def fit(x, n):
    return np.sin(x) / n


n_media = np.average(n, weights=var_n**(-1))

sigma_n = np.sqrt(np.sum(var_n**(-1)))

print("n = ", n_media)
print("sigma_n = ", sigma_n)


"""
# Fit e Grafico

popt, pcov = curve_fit(snell_cartesio, theta_i, theta_r, sigma=sigma_n)
i_hat, n_hat = popt
sigma_i, sigma_n = np.sqrt(pcov.diagonal())
print(i_hat, n_hat, sigma_i, sigma_n)


model = odrpack.Model(snell_cartesio)
data = odrpack.RealData(x, y, sx=sigma_x, sy=sigma_y)
odr = odrpack.ODR(data, model, beta0=(1.0, 1.0))
out = odr.run()
m_hat, q_hat = out.beta
sigma_m, sigma_q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square
ndof = len(x) - 2
print(f"m = {m_hat:.3f} +/- {sigma_m:.3f}")
print(f"q = {q_hat:.3f} +/- {sigma_q:.3f}")
print(q_hat**(-1)," = f")
print(f"Chisquare = {chisq:.1f}/{ndof}")
print("Theoretical f = ",lente_cilindrica())

"""

plt.figure("Indice di rifazione Plexiglas")
plt.xlabel("i [rad]")
plt.ylabel("r [rad]")
plt.errorbar(np.arcsin(sin_i), np.arcsin(sin_r),
             xerr=(sigma_sin_i/np.sqrt(1-sin_i**2)),
             yerr=(sigma_sin_r/np.sqrt(1-sin_r**2)), fmt="o")

x = np.linspace(np.arcsin(sin_i).min(), np.arcsin(sin_i).max(), 10)
plt.plot(x, fit(x, n_media))
plt.grid(which='both', ls='dashed', color='blue')

plt.savefig("grafico.pdf")

plt.show()
