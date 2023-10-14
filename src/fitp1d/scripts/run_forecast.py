import argparse

from mpi4py import MPI
import matplotlib

import fitp1d.plotting
import fitp1d.forecast

matplotlib.use('Agg')


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("InputQMLEFile", help="QMLE file")
    parser.add_argument("InputCovFile", help="Covariance file")
    parser.add_argument("-o", "--output-dir", help="Output directory.")
    parser.add_argument("--slice", action='store_true')
    parser.add_argument(
        "--nlive", help="Number of live points", type=int, default=250)
    parser.add_argument("--slice-steps", type=int, default=20)
    parser.add_argument(
        "--resume", default='resume', help="To resume or no.",
        choices=['resume', 'resume-similar', 'overwrite', 'subfolder']
    )

    return parser


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    args = getParser().parse_args()

    fpl = fitp1d.forecast.P1DLikelihood2(
        args.InputQMLEFile, fname_cov=args.InputCovFile,
    )
    fpl.setFiducial()
    fpl.setMinimizer()

    # No cosmology fit
    for c in fpl.p1dmodel._cosmo_names:
        fpl.fixParam(c, fpl.initial[c])

    log_dir = f"{args.output_dir}/no_cosmology"
    fpl.sampleUnest(
        log_dir, use_slice=args.slice, nlive=args.nlive,
        slice_steps=args.slice_steps, resume=args.resume, mpi_rank=mpi_rank
    )

    # As, mnu
    for c in fpl.fixed_params:
        fpl.releaseParam(c)

    fpl.fixParam("ns", fpl.initial['ns'])
    fpl.fixParam("Ode0", fpl.initial['Ode0'])
    fpl.fixParam("H0", fpl.initial['H0'])
    log_dir = f"{args.output_dir}/as_mnu"
    fpl.sampleUnest(
        log_dir, use_slice=args.slice, nlive=args.nlive,
        slice_steps=args.slice_steps, resume=args.resume, mpi_rank=mpi_rank
    )

    # All
    for c in fpl.fixed_params:
        fpl.releaseParam(c)

    log_dir = f"{args.output_dir}/all_free"
    fpl.sampleUnest(
        log_dir, use_slice=args.slice, nlive=args.nlive,
        slice_steps=args.slice_steps, resume=args.resume, mpi_rank=mpi_rank
    )
