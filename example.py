import py_cosmo_sim_tools as sim
import matplotlib.pyplot as plt
import numpy as np

#print(sim.cosmology.a2z(((1 - sim.cosmology.z2a(200))/625) * 624 + sim.cosmology.z2a(200)))
#exit()

L = 128
NP = 256
nbins = 100
knyq = (np.pi*NP)/L

init = sim.initial_conditions.Initializer(sim.cosmology.get_camb_params(),L)

particles = init.generate(NP)

plt.title(f"NP = {NP}, L = {L}")

kmin = 0
kmax = knyq * 3

sim.power_spectrum.plot_pk(particles,NG = NP, bins = nbins, nfolds=0, show_label = True, kmin = kmin, kmax = kmax)
sim.power_spectrum.plot_pk(particles,NG = NP, bins = nbins, nfolds=1, show_label = True, kmin = kmin, kmax = kmax)
sim.power_spectrum.plot_pk(particles,NG = NP, bins = nbins, nfolds=2, show_label = True, kmin = kmin, kmax = kmax)

plt.xscale('log')
plt.yscale('log')

ymin,ymax = plt.ylim()
plt.vlines([knyq],ymin,ymax,color="black",linestyles="dashed")
plt.ylim(ymin,ymax)

plt.legend()
plt.tight_layout()

plt.show()