import camb
from camb import model
import numba
import numpy as np
from . import cosmology

def log(*args):
    print("IC:",*args)

@numba.jit(nopython=True)
def k1d(NG,RL=None,physical = True):
    ks = np.arange(NG)
    ks[ks >= NG//2] -= NG
    if (physical):
        ks = ks * ((2*np.pi)/RL)
    else:
        ks = ks * ((2*np.pi)/NG)
    return ks

@numba.jit(nopython=True, parallel=True)
def gridk2(NG,RL = None,physical = False):
    ks = k1d(NG,RL,physical)
    #ks = np.arange(NG)
    #ks[ks >= NG//2] -= NG
    #if (physical):
    #    ks = ks * ((2*np.pi)/RL)
    #else:
    #    ks = ks * ((2*np.pi)/NG)
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
    base_mul = -1j * (1/(D))
    for idx in numba.prange(NG):
        i = ks[idx]
        for jdx in range(NG):
            j = ks[jdx]
            for kdx in range(NG):
                k = ks[kdx]
                if (i*i + j*j + k*k) == 0:
                    continue
                rho = rho_k[idx,jdx,kdx]
                mul = base_mul * rho
                sx[idx,jdx,kdx] = mul * (i / (i*i + j*j + k*k))
                sy[idx,jdx,kdx] = mul * (j / (i*i + j*j + k*k))
                sz[idx,jdx,kdx] = mul * (k / (i*i + j*j + k*k))

    return sx,sy,sz

@numba.jit(nopython=True, parallel=True)
def generate_x(NG,sx,sy,sz,D):
    position = np.zeros((NG*NG*NG,3),dtype=float)
    mul = 1.0/float(NG)
    for i in numba.prange(NG):
        for j in range(NG):
            for k in range(NG):
                idx = (i * NG + j) * NG + k
                position[idx][0] = (i + D * sx[i,j,k]) * mul
                position[idx][1] = (j + D * sy[i,j,k]) * mul
                position[idx][2] = (k + D * sz[i,j,k]) * mul

    return position

@numba.jit(nopython=True, parallel=True)
def generate_v(NG,sx,sy,sz,Ddot,a):
    position = np.zeros((NG*NG*NG,3),dtype=float)
    mul = (a * Ddot) / NG
    for i in numba.prange(NG):
        for j in range(NG):
            for k in range(NG):
                idx = (i * NG + j) * NG + k
                position[idx][0] = sx[i,j,k] * mul
                position[idx][1] = sy[i,j,k] * mul
                position[idx][2] = sz[i,j,k] * mul
    return position

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
        log("Getting power spectrum from CAMB")
        ks = gridk2(NP,self.L,physical=True)
        self.camb_params.set_matter_power(redshifts=[z], kmax = np.max(ks))
        results = camb.get_results(self.camb_params)
        self.camb_params.NonLinear = model.NonLinear_none
        out = results.get_matter_power_interpolator().P(z,ks.flatten()[1:])
        return np.concatenate((np.array([0.0]),out)).reshape((NP,NP,NP)) * ((NP*NP*NP)/(self.L*self.L*self.L))

    def generate_fourier_amplitudes(self, NP : int, z : float):
        log("Generating fourier amplitudes")
        return np.fft.fftn(np.random.normal(loc=0,scale=1.0,size=NP*NP*NP).reshape((NP,NP,NP))) * np.sqrt(self.get_pk(NP,z))

    def generate(self, NP : int, z_x : float = 200, z_v = None, dt : float = (1 - (1/(1+200)))/625, generate_velocity = True):
        a_x = cosmology.z2a(z_x)
        if (z_v is None):
            a_v = a_x - (dt/2)
        else:
            a_v = cosmology.z2a(z_v)
        log("Generating ICs with a_x =",a_x,"a_v =",a_v,"NP =",NP,"L =",self.L)
        rho_k = self.generate_fourier_amplitudes(NP,cosmology.a2z(a_x))
        results = camb.get_results(self.camb_params)
        delta = cosmology.D(a_x,results)
        dot_delta = cosmology.dotD(a_v,results)
        log("Moving particles with delta =",delta,"dot_delta =",dot_delta)
        sl,sm,sn = generate_s(NP,self.L,delta,rho_k)
        log("Doing backwards FFTs")
        sx,sy,sz = np.real(np.fft.ifftn(sl)), np.real(np.fft.ifftn(sm)), np.real(np.fft.ifftn(sn))
        log("Displacing particles (position)")
        x = generate_x(NP,sx,sy,sz,delta)
        if (generate_velocity):
            log("Displacing particles (velocity)")
            v = generate_v(NP,sx,sy,sz,dot_delta,a_v)
        else:
            v = None
        log("Done")
        return Particles(x,v,results,self.L,cosmology.a2z(a_x),cosmology.a2z(a_v))

class SineInitializer(Initializer):
    def __init__(self, camb_params : camb.CAMBparams, L : float, k : float, amp : float = 10.0):
        super().__init__(camb_params,L)
        self.k = k
        self.amp = amp

    def get_pk(self, NP : int, z : float):
        log("Generating delta function power spectrum")
        ks = k1d(NP,self.L)
        keep = 0
        kpole = 0
        for idx,i in enumerate(ks):
            if (i >= self.k):
                keep = idx
                kpole = i
                break
        log(f"Using k = ({kpole},0,0)")
        tmp = np.zeros((NP,NP,NP),dtype=float)
        tmp[keep,0,0] = self.amp
        tmp[-keep,0,0] = self.amp
        self.kpole = kpole
        return tmp