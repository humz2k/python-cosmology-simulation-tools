import matplotlib.pyplot as plt
import numpy as np

def plot_pretty(dpi=150,fontsize=15):
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    plt.rc('legend',fontsize=5)
    plt.rc('text',usetex=False)
    plt.rcParams['figure.figsize'] = [5, 2]

def plot_vline(x,**kwargs):
    ymin,ymax = plt.ylim()
    plt.vlines([x],ymin,ymax,**kwargs)
    plt.ylim(ymin,ymax)

def plot_nyquist(NG,L,multiplier = 1,**kwargs):
    knyq = (np.pi*NG)/L
    ymin,ymax = plt.ylim()
    plt.vlines([knyq * multiplier],ymin,ymax,**kwargs)
    plt.ylim(ymin,ymax)