# import os, sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import odr

# Misure
p = np.array([16., 13.6, 13., 14.8, 18.5, 16.8, 14.9])  # [cm]
q = np.array([14., 16.2, 18., 15.7, 12.5, 13.5, 16])  # [cm]

x = p**(-1)
y = q**(-1)

# Errori
sigma_p_q = np.full(p.shape, 0.3)  # [cm]
sigma_x = sigma_p_q/(p**2)  # [cm]
sigma_y = sigma_p_q/(q**2)  # [cm]


# Funzioni
def lente_cilindrica(r=3.75, n=1.33):
    f_inverso = (2/r)*((n-1)/n)     # f_inverso = 1/f
    return f_inverso**(-1)


def fit_model(pars, x):
    return pars[0] * x + pars[1]


model = odr.Model(fit_model)
data = odr.RealData(x, y, sx=sigma_x, sy=sigma_y)
odr = odr.ODR(data, model, beta0=(1.0, 1.0))
out = odr.run()
m_hat, q_hat = out.beta
sigma_m, sigma_q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square
ndof = len(x) - 2
print(f"m = {m_hat:.3f} +/- {sigma_m:.3f}")
print(f"q = {q_hat:.3f} +/- {sigma_q:.3f}")
print(q_hat**(-1), " = f")
print(f"Chisquare = {chisq:.1f}/{ndof}")
print("Theoretical f = ", lente_cilindrica())

# Grafico
grafico = plt.figure('Distanza schermo')
grafico.add_axes((0.15, 0.1, 0.8, 0.8))
plt.errorbar(x, y, yerr=sigma_y, xerr=sigma_x, fmt='.')

plt.xlabel(r"$\frac{1}{q} [\mathrm{cm}^{-1}]$",
           fontdict={"usetex": True}, fontsize=12)
plt.ylabel(r"$\frac{1}{p} [\mathrm{cm}^{-1}]$",
           fontdict={"usetex": True}, fontsize=12)

plt.text(0.054, 0.062, fr"$\chi^2 /\nu:{round(chisq, 1)}/{ndof}$",
         fontdict={"usetex": True}, fontsize=20)

x = np.linspace(x.min(), x.max(), 10)
plt.plot(x, fit_model(out.beta, x))
plt.grid(which='both', ls='dashed', color='blue')

print(lente_cilindrica())

plt.savefig('lente_cilindrica.pdf')
plt.show()
