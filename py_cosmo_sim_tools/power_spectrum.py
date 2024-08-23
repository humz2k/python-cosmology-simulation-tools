from . import initial_conditions

import numpy as np
import numba
import matplotlib.pyplot as plt

@numba.jit(nopython=True, parallel=True)
def deconvolve_cic(kfield):
    N = kfield.shape[0]

    for l in numba.prange(N):
        for m in range(N):
            for n in range(N // 2 + 1):
                lk = l if l < N // 2 else l - N
                mk = m if m < N // 2 else m - N
                nk = n

                wx = 1.0 if lk == 0 else np.sinc(lk / N)
                wy = 1.0 if mk == 0 else np.sinc(mk / N)
                wz = 1.0 if nk == 0 else np.sinc(nk / N)
                w = (wx * wy * wz) ** 2
                kfield[l, m, n] /= w

@numba.jit(nopython=True)
def radial_binning(kfield1, kfield2, L, nbins, nfolds = 0, kmin = None, kmax = None):
    N = kfield1.shape[0]
    if (kmin is None):
        kmin = 1.0
    if (kmax is None):
        kmax = np.sqrt(3) * N / 2
    if nbins == 0:
        nbins = int(np.ceil(kmax - kmin))
    dk = (kmax - kmin) / nbins

    # Fields to store power spectrum
    ps = np.zeros(nbins, dtype=np.float64)
    pn = np.zeros(nbins, dtype=np.int64)
    pk = np.zeros(nbins, dtype=np.float64)
    pk2 = np.zeros(nbins, dtype=np.float64)

    kfac = 2 * np.pi / (L / (2 ** nfolds))

    for l in range(N):
        for m in range(N):
            for n in range(N // 2 + 1):
                lk = l if l < N // 2 else l - N
                mk = m if m < N // 2 else m - N
                nk = n

                k = np.sqrt(lk**2 + mk**2 + nk**2) * kfac
                if k == 0:
                    continue
                k_index = int((np.sqrt(lk**2 + mk**2 + nk**2) - kmin) / dk)

                if k_index >= 0 and k_index < nbins:
                    v2 = kfield1[l, m, n] * np.conjugate(kfield2[l, m, n])

                    if n == 0:
                        ps[k_index] += np.real(v2)
                        pn[k_index] += 1
                        pk[k_index] += k
                        pk2[k_index] += k**2
                    else:
                        ps[k_index] += np.real(v2 + np.conjugate(v2))
                        pn[k_index] += 2
                        pk[k_index] += 2 * k
                        pk2[k_index] += 2 * k**2

    mask = pn > 0
    psm = np.empty((np.sum(mask), 4))
    j = 0
    for i in range(nbins):
        if mask[i]:
            psm[j, 0] = pk[i] / pn[i]
            psm[j, 1] = ps[i] / pn[i] / L**3
            psm[j, 2] = pn[i]
            psm[j, 3] = pk2[i] / pn[i] - (pk[i] / pn[i]) ** 2
            if psm[j, 3] < 0:
                psm[j, 3] = 0
            j += 1

    return psm


def compute_pk_auto(delta, nbins, L, deconvolve_CIC=True, nfolds = 0, **kwargs):
    N = delta.shape[0]
    delta = np.fft.rfftn(delta) * (L / N) ** 3
    if deconvolve_CIC:
        deconvolve_cic(delta)
    return radial_binning(delta, delta, L, nbins, nfolds = nfolds, **kwargs)


@numba.jit(nopython=True)
def _cic3d(grid, pos3d, resolution, shift = False):
    shift_amount = 0.5 if shift else 0.0
    x = np.empty(3, dtype=np.float32)
    for i in range(len(pos3d)):
        x[:] = pos3d[i] * resolution

        # we want x to be the lower left corner of the cube which we are depositing
        x[0] = np.fmod(resolution[0] + x[0] - shift_amount, resolution[0])
        x[1] = np.fmod(resolution[1] + x[1] - shift_amount, resolution[1])
        x[2] = np.fmod(resolution[2] + x[2] - shift_amount, resolution[2])

        ix = np.uint32(x[0])
        iy = np.uint32(x[1])
        iz = np.uint32(x[2])

        ix1 = (ix + 1) % resolution[0]
        iy1 = (iy + 1) % resolution[1]
        iz1 = (iz + 1) % resolution[2]

        dx = x[0] - ix
        dy = x[1] - iy
        dz = x[2] - iz

        ix %= resolution[0]
        iy %= resolution[1]
        iz %= resolution[2]

        tx = 1 - dx
        ty = 1 - dy
        tz = 1 - dz

        grid[ix, iy, iz] += tx * ty * tz
        grid[ix, iy, iz1] += tx * ty * dz
        grid[ix, iy1, iz] += tx * dy * tz
        grid[ix, iy1, iz1] += tx * dy * dz

        grid[ix1, iy, iz] += dx * ty * tz
        grid[ix1, iy, iz1] += dx * ty * dz
        grid[ix1, iy1, iz] += dx * dy * tz
        grid[ix1, iy1, iz1] += dx * dy * dz


def cic3d(pos3d, resolution, *, normalize=True, shift = False):
    if not hasattr(resolution, "__len__"):
        resolution = np.array([resolution, resolution, resolution])
    grid = np.zeros(resolution, dtype=np.float64)
    _cic3d(grid, pos3d, resolution, shift = shift)

    if normalize:
        grid /= np.mean(grid)

    return grid

def fold(particles, nfolds = 0):
    tmp = np.fmod(particles * (2 ** nfolds) + 1, 1)
    return tmp

def calc_pk(particles : initial_conditions.Particles, NG, bins = 50, nfolds = 0, deconvolve_CIC=True, shift = False, kmin = None, kmax = None, **kwargs):
    L = particles.L
    if (kmin is not None):
        kmin = kmin / (2 * np.pi / (L / (2 ** nfolds)))
    if (kmax is not None):
        kmax = kmax / (2 * np.pi / (L / (2 ** nfolds)))
    density = cic3d(fold(particles.position,nfolds),NG,shift = shift)
    pk = compute_pk_auto(density - 1, bins, L, deconvolve_CIC=deconvolve_CIC, nfolds = nfolds, kmin = kmin, kmax = kmax)
    return {"k": pk[:,0], "P_0": pk[:,1]}

def plot_pk(particles : initial_conditions.Particles, NG, bins = 50, nfolds = 0, deconvolve_CIC=True, shift = False, kmin = None, kmax = None, show_label = True, **kwargs):
    pk = calc_pk(particles,NG,bins=bins,nfolds=nfolds,deconvolve_CIC=deconvolve_CIC,shift=shift,kmin=kmin,kmax=kmax)
    if show_label:
        plt.plot(pk['k'],pk['P_0'],**kwargs,label="NG = " + str(NG) + ", nfolds = " + str(nfolds) + ", grid_shift = " + ("true" if shift else "false"))
    else:
        plt.plot(pk['k'],pk['P_0'],**kwargs)
    return pk

def lerp_pk(particles : initial_conditions.Particles, NG, ks, bins = 50, nfolds = 0, deconvolve_CIC=True, shift = False, kmin = None, kmax = None, **kwargs):
    pk = calc_pk(particles,NG,bins=bins,nfolds=nfolds,deconvolve_CIC=deconvolve_CIC,shift=shift,kmin=kmin,kmax=kmax)
    return {'k': ks,'P_0': np.interp(ks,pk['k'],pk['P_0'])}