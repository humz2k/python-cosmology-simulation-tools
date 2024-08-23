import py_cosmo_sim_tools as sim
import matplotlib.pyplot as plt
import numpy as np

L = 16
NP = 32
nbins = 20
knyq = (np.pi*NP)/L

init_k = 3.15

init = sim.initial_conditions.SineInitializer(sim.cosmology.get_camb_params(),L,init_k,100)

particles = init.generate(NP)

plt.title("k = (" + str(round(init.kpole,2)) + ",0,0)")

sim.power_spectrum.plot_pk(particles,NG = NP, bins = nbins, nfolds=0, show_label = True, shift = True)
sim.power_spectrum.plot_pk(particles,NG = NP, bins = nbins, nfolds=1, show_label = True, shift = True)
sim.power_spectrum.plot_pk(particles,NG = NP * 2, bins = nbins, nfolds=0, show_label = True, shift = True)

ymin,ymax = plt.ylim()
plt.vlines([knyq],ymin,ymax,color="black",linestyles="dashed")
plt.ylim(ymin,ymax)

plt.legend()
plt.tight_layout()

plt.show()