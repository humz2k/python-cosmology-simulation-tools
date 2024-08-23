import camb
from camb import model
import numba
import numpy as np
from . import cosmology

@numba.jit(nopython=True)
def k1d(NG,RL):
    ks = np.arange(NG)
    ks[ks >= NG//2] -= NG
    ks = ks * ((2*np.pi)/RL)
    return ks

@numba.jit(nopython=True, parallel=True)
def gridk2(NG,RL = None,physical = False):
    ks = np.arange(NG)
    ks[ks >= NG//2] -= NG
    if (physical):
        ks = ks * ((2*np.pi)/RL)
    else:
        ks = ks * ((2*np.pi)/NG)
    grid_ks = np.zeros((NG,NG,NG))
    for idx in numba.prange(NG):
        i = ks[idx]
        for jdx in range(NG):
            j = ks[jdx]
            for kdx in range(NG):
                k = ks[kdx]
                grid_ks[idx,jdx,kdx] = np.sqrt(i*i + j*j + k*k)
    return grid_ks

@numba.jit(nopython=True, parallel=True)
def generate_s(NG, RL, D, rho_k):
    ks = np.arange(NG)
    ks[ks >= NG//2] -= NG
    ks = ks * ((2*np.pi)/NG)

    sx = np.zeros((NG,NG,NG),dtype=np.complex128)
    sy = np.zeros((NG,NG,NG),dtype=np.complex128)
    sz = np.zeros((NG,NG,NG),dtype=np.complex128)
    for idx in numba.prange(NG):
        i = ks[idx]
        for jdx in range(NG):
            j = ks[jdx]
            for kdx in range(NG):
                k = ks[kdx]
                if (i*i + j*j + k*k) == 0:
                    continue
                rho = rho_k[idx,jdx,kdx]
                sx[idx,jdx,kdx] = -1j * (1/(D)) * (i / (i*i + j*j + k*k)) * rho
                sy[idx,jdx,kdx] = -1j * (1/(D)) * (j / (i*i + j*j + k*k)) * rho
                sz[idx,jdx,kdx] = -1j * (1/(D)) * (k / (i*i + j*j + k*k)) * rho

    return sx,sy,sz

@numba.jit(nopython=True, parallel=True)
def generate_x(NG,L,sx,sy,sz,D):
    xx = np.zeros((NG,NG,NG),dtype=float)
    yy = np.zeros((NG,NG,NG),dtype=float)
    zz = np.zeros((NG,NG,NG),dtype=float)
    for i in numba.prange(NG):
        for j in range(NG):
            for k in range(NG):
                xx[i,j,k] = i + D * sx[i,j,k]
                yy[i,j,k] = j + D * sy[i,j,k]
                zz[i,j,k] = k + D * sz[i,j,k]

    return np.column_stack((xx.flatten(),yy.flatten(),zz.flatten()))

@numba.jit(nopython=True, parallel=True)
def generate_v(NG,L,sx,sy,sz,Ddot,a):
    xx = np.zeros((NG,NG,NG),dtype=float)
    yy = np.zeros((NG,NG,NG),dtype=float)
    zz = np.zeros((NG,NG,NG),dtype=float)
    for i in numba.prange(NG):
        for j in range(NG):
            for k in range(NG):
                xx[i,j,k] = a * Ddot * sx[i,j,k]
                yy[i,j,k] = a * Ddot * sy[i,j,k]
                zz[i,j,k] = a * Ddot * sz[i,j,k]

    return np.column_stack((xx.flatten(),yy.flatten(),zz.flatten()))

class Particles:
    def __init__(self, position, velocity, camb_data : camb.CAMBdata, L, z_x_ini, z_v_ini):
        self.position = position
        self.velocity = velocity
        self.camb_data = camb_data
        self.L = L
        self.z_x_ini = z_x_ini
        self.z_v_ini = z_v_ini
        self.size = self.position.shape[0]

class Initializer:
    def __init__(self, camb_params : camb.CAMBparams, L : float):
        self.camb_params : camb.CAMBparams = camb_params
        self.L : float = L

    def get_pk(self, NP : int, z : float):
        ks = gridk2(NP,self.L,physical=True)
        self.camb_params.set_matter_power(redshifts=[z], kmax = np.max(ks))
        results = camb.get_results(self.camb_params)
        self.camb_params.NonLinear = model.NonLinear_none
        out = results.get_matter_power_interpolator().P(z,ks.flatten()[1:])
        return np.concatenate((np.array([0.0]),out)).reshape((NP,NP,NP)) * ((NP*NP*NP)/(self.L*self.L*self.L))

    def generate_fourier_amplitudes(self, NP : int, z : float):
        return np.fft.fftn(np.random.normal(loc=0,scale=1.0,size=NP*NP*NP).reshape((NP,NP,NP))) * np.sqrt(self.get_pk(NP,z))

    def generate(self, NP : int, z_x : float = 200, z_v = None, dt : float = (1 - (1/(1+200)))/625):
        a_x = cosmology.z2a(z_x)
        if (z_v is None):
            a_v = a_x - (dt/2)
        else:
            a_v = cosmology.z2a(z_v)
        rho_k = self.generate_fourier_amplitudes(NP,cosmology.a2z(a_x))
        results = camb.get_results(self.camb_params)
        delta = cosmology.D(a_x,results)
        dot_delta = cosmology.dotD(a_v,results)
        sl,sm,sn = generate_s(NP,self.L,delta,rho_k)
        sx,sy,sz = np.real(np.fft.ifftn(sl)), np.real(np.fft.ifftn(sm)), np.real(np.fft.ifftn(sn))
        x = generate_x(NP,self.L,sx,sy,sz,delta) / NP
        v = generate_v(NP,self.L,sx,sy,sz,dot_delta,a_v) / NP
        return Particles(x,v,results,self.L,cosmology.a2z(a_x),cosmology.a2z(a_v))

class SineInitializer(Initializer):
    def __init__(self, camb_params : camb.CAMBparams, L : float, k : float, amp : float = 10.0):
        super().__init__(camb_params,L)
        self.k = k
        self.amp = amp

    def get_pk(self, NP : int, z : float):
        ks = k1d(NP,self.L)
        keep = 0
        kpole = 0
        for idx,i in enumerate(ks):
            if (i >= self.k):
                keep = idx
                kpole = i
                break

        tmp = np.zeros((NP,NP,NP),dtype=float)
        tmp[keep,0,0] = self.amp
        tmp[-keep,0,0] = self.amp
        self.kpole = kpole
        return tmp