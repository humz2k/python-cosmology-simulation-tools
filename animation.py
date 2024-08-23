import py_cosmo_sim_tools as sim
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import *

plot_pretty(400,fontsize=5)

L = 128
NP = 256
knyq = (np.pi*NP)/L

os.system("rm -rf frames2")
os.system("mkdir frames2")

ks = sim.initial_conditions.k1d(NP,L,physical=True)
for idx,k in enumerate(ks[2:][::2]):
    if (k < 0):
        break
    init = sim.initial_conditions.SineInitializer(sim.cosmology.get_camb_params(),L,k,1000)
    particles = init.generate(NP, generate_velocity = False)

    nbins = 50
    pk_np_ng = sim.power_spectrum.calc_pk(particles,NG=NP,bins=nbins,nfolds=0,shift=True)
    pk_np_ng_2 = sim.power_spectrum.calc_pk(particles,NG=NP//2,bins=nbins,nfolds=0,shift=True)
    pk_np_ng_2_fold_1 = sim.power_spectrum.calc_pk(particles,NG=NP//2,bins=nbins,nfolds=1,shift=True)

    #normalize = np.max(pk_np_ng['P_0'])
    #tmp = np.max(pk_np_ng_2['P_0'])
    #if (tmp > normalize):
    #    normalize = tmp
    #tmp = np.max(pk_np_ng_2_fold_1['P_0'])
    #if (tmp > normalize):
    #    normalize = tmp

    plt.title(f"NP={NP}, k=({round(init.kpole,2)},0,0)")
    plt.plot(pk_np_ng["k"],pk_np_ng["P_0"]/np.max(pk_np_ng["P_0"]),label=f"NG={NP}, nfolds={0}, shift=True")
    plt.plot(pk_np_ng_2["k"],pk_np_ng_2["P_0"]/np.max(pk_np_ng_2["P_0"]),label=f"NG={NP//2}, nfolds={0}, shift=True")
    plt.plot(pk_np_ng_2_fold_1["k"],pk_np_ng_2_fold_1["P_0"]/np.max(pk_np_ng_2_fold_1["P_0"]),label=f"NG={NP//2}, nfolds={1}, shift=True")
    plot_nyquist(NP//2,L,color="black",linestyles="dashed",label=f"nyq(NG={NP//2},L={L})")
    plot_nyquist(NP//2,L//2,color="black",linestyles="dotted",label=f"nyq(NG={NP//2},L={L//2} \\&\\& NG={NP},L={L})")
    plot_vline(init.kpole,color="red",linestyles="dashed",alpha=0.5,label=f"k={round(init.kpole,2)}")
    plt.legend()
    #plt.axes().get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.savefig(f"frames2/frame{idx}.jpg")
    plt.close()