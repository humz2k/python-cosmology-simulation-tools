import numpy as np
import scipy.integrate
import scipy.misc
import camb

def z2a(z : float) -> float:
    return (1.0/(1.0+z))

def a2z(a : float) -> float:
    return (1.0/a) - 1.0

def growth_factor(a, pars : camb.CAMBdata, k = 1):
    return pars.get_redshift_evolution(k,a2z(a),vars='delta_tot').flatten() / pars.get_redshift_evolution(k,0,vars='delta_tot').flatten()

def get_H(a, params : camb.CAMBdata):
    w_de = params.get_dark_energy_rho_w(1/(1+200))[1]
    wa_de = 0

    f_nu_massless = (params.grhornomass / params.grhor) * 7.0 / 8.0 * pow(4.0 / 11.0, 4.0 / 3.0)
    omega_radiation = params.get_Omega("photon")

    H = np.sqrt((params.get_Omega("cdm") + params.get_Omega("baryon")) / pow(a, 3.0) +
             (1.0 + f_nu_massless) * omega_radiation /
                 pow(a, 4.0) +
             params.get_Omega("nu") +
             (1.0 - (params.get_Omega("cdm") + params.get_Omega("baryon") + params.get_Omega("nu")) -
              (1.0 + f_nu_massless) * omega_radiation *
                  pow(a, (-3.0 * (1.0 + w_de + wa_de))) *
                  np.exp(-3.0 * wa_de * (1.0 - a))))
    return H

def growth_factor_deriv(a, pars : camb.CAMBdata):
    return scipy.misc.derivative(growth_factor,a,0.001,args=(pars,),order=3) * a * get_H(a,pars)

def get_camb_params(OMEGA_CDM : float = 0.22, DEUT : float = 0.02258, HUBBLE = 0.71, SS8 = 0.8, NS = 0.963, OMEGA_K = 0, N_EFF_MASSLESS = 3.04, N_EFF_MASSIVE = 0.0, T_CMB = 2.726):
    h2 = HUBBLE**2
    pars = camb.CAMBparams()
    pars.set_cosmology(H0 = HUBBLE * 100, ombh2 = DEUT, omch2 = OMEGA_CDM * h2, omk = OMEGA_K, num_massive_neutrinos=1, nnu = N_EFF_MASSLESS, standard_neutrino_neff = N_EFF_MASSIVE, neutrino_hierarchy='normal', TCMB=T_CMB)
    pars.InitPower.set_params(ns = NS)
    return pars

def D(a, pars):
    return growth_factor(a, pars)[0]

def dotD(a, pars):
    return growth_factor_deriv(a, pars)[0]

if __name__ == "__main__":
    params = camb.get_results(get_camb_params())
    print(D((1/(1+200)),params))
    print(dotD((1/(1+200)),params))
