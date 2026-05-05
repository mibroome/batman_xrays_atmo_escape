from mpi4py import MPI
import numpy as np
import batman
import xt_fns as xf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------------------------
# Stellar / orbital parameters — fixed for all Z and E
# ---------------------------------------------------------------------------
Rs   = 0.805   # stellar radius (Rsun)
Ms   = 0.846   # stellar mass (Msun)
T    = 0.34    # coronal temperature (keV)
aRs  = 8.2830  # semi-major axis in stellar radii
inc  = 85.580  # orbital inclination (degrees)

Rp_surf = 8135789600.0  # planet surface radius (cm)

# Emission scale-height quantities are E-independent; compute once on all ranks
He     = xf.HfromT(T, Rs, Ms)
he     = xf.hfromH(He)
Nscale = 6
Rx      = xf.calcRx(he, Nscale)
intVals = xf.get_intVals(he, Nscale)

# Phase grid is also E-independent
phaSt, phaFi = 0.9, 1.1
numBins      = 500
binPhases    = np.linspace(phaSt, phaFi, numBins)

# ---------------------------------------------------------------------------
# Main loop over metallicity
# ---------------------------------------------------------------------------
# for Z in [1, 5, 10]:
#     for enhancement in [1,4]:
for Z in [3]:
    # All ranks load the (read-only) profile arrays — fast numpy I/O
    profiles_array       = np.load(f'../batman_xrays_atmo_escape/tau_profiles/tau_HD189_{Z}Z_soft.npy',
                                allow_pickle=True)
    # profiles_array_shock = np.load(f'../batman_xrays_atmo_escape/tau_profiles/tau_HD189_{Z}Z_shock_enh{enhancement}.npy',
    #                             allow_pickle=True)

    n_profiles   = len(profiles_array)
    my_indices   = np.array_split(np.arange(n_profiles), size)[rank]

    my_results = []

    for idx in my_indices:
        profile       = profiles_array[idx]
        # profile_shock = profiles_array_shock[idx]

        E           = profile['E']
        tau_array     = profile['taus']
        tau_array_bow = profile['taus']

        print(f'rank {rank:02d} | Z={Z:2d} | E={E:.1f} eV', flush=True)

        Rmax  = max(profile['rs']) / Rp_surf #messed up and b_array for profile_shock is in units of Rp, so make sure this is always profile
        Rp    = Rp_surf * Rmax
        RpRs  = Rp / (Rs * 6.957e10)

        params               = batman.TransitParams()
        params.t0            = 1
        params.per           = 1
        params.rp            = RpRs * Rx
        params.a             = aRs * Rx
        params.inc           = inc
        params.ecc           = 0.
        params.w             = 90.
        params.limb_dark     = "custom"
        params.u             = [0] * 6
        params.u[0]          = intVals
        params.tau           = tau_array
        params.tau2          = tau_array_bow

        mod   = batman.TransitModel(params, binPhases, fac=5e-4)
        flux1 = mod.light_curve(params)

        my_results.append({'E': E, 'binPhases': binPhases, 'transmission_frac': flux1})

    # ---- gather and save on rank 0 -----------------------------------------
    all_results = comm.gather(my_results, root=0)

    if rank == 0:
        flat = [item for sublist in all_results for item in sublist]
        flat.sort(key=lambda d: d['E'],reverse=True)
        np.save(f'../copy/transit_profiles/All_wavelengths/HD189_{Z}Z_soft.npy',
                np.array(flat, dtype=object), allow_pickle=True)
        print(f'\n[rank 0] Z={Z} saved ({len(flat)} profiles)', flush=True)

    comm.Barrier()
