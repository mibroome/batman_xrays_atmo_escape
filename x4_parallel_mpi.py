from mpi4py import MPI
import numpy as np
from wind_ae.wrapper.relax_wrapper import wind_simulation as wind_sim
from wind_ae.wrapper.wrapper_utils import constants as const
from scipy.interpolate import Akima1DInterpolator
from shock_tau_funcs import LOS_slice_calculator, bow_tau, total_tau_calculator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Z_values     = [1, 5, 10]
enhancements = [1, 4]

# ---------------------------------------------------------------------------
# Load the stellar spectrum once on each rank.
# Any Z works here — the spectrum is stellar, not atmosphere-dependent.
# ---------------------------------------------------------------------------
_spec_sim = wind_sim()
_spec_sim.load_uservars('saves/hd189_2-24-26/HD189_1Z.csv')
_spec_sim.load_spectrum()
spec_data   = _spec_sim.spectrum.data
E_array_all = spec_data['E'].to_numpy() / const.eV
eV_mask     = (E_array_all > 166) & (E_array_all < 2000)
E_to_process = E_array_all[eV_mask]

# ---------------------------------------------------------------------------
# Outer loops: Z then enhancement.
# Models are loaded on every rank — this is fast (file reads) and avoids
# any inter-rank communication for large arrays.
# The E loop is split across ranks.
# ---------------------------------------------------------------------------
for Z in Z_values:

    simwind = wind_sim()
    simwind.load_uservars(f'saves/hd189_2-24-26/HD189_{Z}Z.csv')

    simbreeze = wind_sim()
    simbreeze.load_uservars(f'saves/hd189_2-24-26/breezes/HD189_{Z}Z_breeze_0.99.csv')

    # LOS geometry is enhancement-independent, compute once per Z.
    # NOTE: x4_1Z.py called LOS_slice_calculator(Z, plot=False) which misses
    # the leading `sim` argument — passing a fresh wind_sim() here instead.
    sim_shock = wind_sim()
    yslices, x_left, x_right = LOS_slice_calculator(sim_shock, Z, plot=False)

    Rmax = simwind.windsoln.Rmax

    for enhancement in enhancements:

        # --- distribute E indices round-trip across ranks -------------------
        my_E_indices = np.array_split(np.arange(len(E_to_process)), size)[rank]
        my_E_values  = E_to_process[my_E_indices]

        my_results = []

        for E in my_E_values:
            print(f'rank {rank:02d} | Z={Z:2d} enh={enhancement} | E={E:.1f} eV', flush=True)

            b_array, tau_b, tau_b_species = bow_tau(
                simwind, simbreeze,
                yslices, x_left, x_right,
                E_phot=E, ds=0.01, enhancement=enhancement
            )

            # ---- same cleaning / stitching logic as original ---------------
            tau_copy = np.copy(tau_b)
            if Z < 10:
                badmin      = tau_copy[np.searchsorted(b_array, 3.6)] / 5
                delete_mask = (tau_copy < badmin) & (b_array < 3.5)
            else:
                badmin      = tau_copy[np.searchsorted(b_array, 2.0)] / 5
                delete_mask = (tau_copy < badmin) & (b_array < 2.0)

            tau_b_clean = np.delete(tau_copy, delete_mask)
            tau_b_clean[tau_b_clean < badmin / 100] = badmin / 100
            b_arr_clean = np.delete(b_array, delete_mask)

            b_array_og, tau_b_og = total_tau_calculator(
                simwind.windsoln, E, simbreeze.windsoln.species_list
            )

            # prepend spherically-symmetric inner solution
            inner_mask    = b_array_og < min(b_arr_clean)
            b_combo       = np.concatenate([b_array_og[inner_mask], b_arr_clean])
            tau_combo     = np.concatenate([tau_b_og[inner_mask],   tau_b_clean])

            # prepend planet (b < min(b_combo))
            b_planet   = np.linspace(0, min(b_combo) - 0.1, 10)
            tau_planet = np.full(10, np.nanmax(tau_combo) if np.nanmax(tau_combo) > 1 else 10.0)
            b_combo    = np.append(b_planet,   b_combo)
            tau_combo  = np.append(tau_planet, tau_combo)

            # append low-opacity tail beyond shock
            b_tail   = np.linspace(max(b_combo) + 0.1, Rmax, 10)
            tau_tail = np.full(10, np.nanmin(tau_combo) / 100)
            b_combo  = np.append(b_combo,   b_tail)
            tau_combo = np.append(tau_combo, tau_tail)

            sort_idx  = np.argsort(b_combo)
            b_combo   = b_combo[sort_idx]
            tau_combo = tau_combo[sort_idx]

            f_interp     = Akima1DInterpolator(b_combo, tau_combo)
            b_final      = np.linspace(0, Rmax, 100)
            tau_final    = f_interp(b_final)

            my_results.append({'E': E, 'rs': b_final, 'taus': tau_final})

        # ---- gather all results to rank 0 ----------------------------------
        all_results = comm.gather(my_results, root=0)

        if rank == 0:
            flat = [item for sublist in all_results for item in sublist]
            flat.sort(key=lambda d: d['E'])

            out_stem = (f'../batman_xrays_atmo_escape/tau_profiles/'
                        f'tau_HD189_{Z}Z_shock_enh{enhancement}')

            # write CSV (one row per energy: E, b_array x100, tau x100)
            with open(out_stem + '.csv', 'w') as fh:
                for item in flat:
                    row = np.concatenate(([item['E']], item['rs'], item['taus']))
                    fh.write(','.join(f'{v}' for v in row) + '\n')

            np.save(out_stem + '.npy', np.array(flat, dtype=object))
            print(f'\n[rank 0] saved Z={Z} enhancement={enhancement} '
                  f'({len(flat)} energies)', flush=True)

        # all ranks wait before moving to next (Z, enhancement) pair
        comm.Barrier()
