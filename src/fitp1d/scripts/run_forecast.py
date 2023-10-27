import argparse

from mpi4py import MPI
import matplotlib
import numpy as np

import fitp1d.plotting
import fitp1d.forecast

matplotlib.use('Agg')


def getParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("InputQMLEFile", help="QMLE file")
    parser.add_argument("InputCovFile", help="Covariance file")
    parser.add_argument("-o", "--output-dir", help="Output directory.")
    parser.add_argument("--sample", help="Runs sampler")
    parser.add_argument("--slice", action='store_true')
    parser.add_argument(
        "--nlive", help="Number of live points", type=int, default=128)
    parser.add_argument("--slice-steps", type=int, default=16)
    parser.add_argument(
        "--resume", default='resume', help="To resume or no.",
        choices=['resume', 'resume-similar', 'overwrite', 'subfolder']
    )

    return parser


def oneSample(fpl, label, fixed_params, args, comm, fix_cosmo=False):
    mpi_rank = comm.Get_rank()

    fpl.releaseAll()
    fpl.setMinimizer()
    if fix_cosmo:
        fpl.fixedCosmology()
    for c in fixed_params:
        fpl.fixParam(c, fpl.initial[c])

    fpl.fit()
    covariance = fpl._mini.covariance.copy()
    np.save(f"{args.output_dir}/covariance_{label}", covariance.to_dict())

    if args.sample:
        log_dir = f"{args.output_dir}/{label}"
        fpl.sampleUnest(
            log_dir, use_slice=args.slice, nlive=args.nlive,
            slice_steps=args.slice_steps, resume=args.resume, mpi_rank=mpi_rank
        )

    comm.Barrier()


def main():
    comm = MPI.COMM_WORLD

    args = getParser().parse_args()

    fpl = fitp1d.forecast.P1DLikelihood2(
        args.InputQMLEFile, fname_cov=args.InputCovFile,
    )
    fpl.setFiducial()

    oneSample(
        fpl, "no_cosmology", [], args, comm, fix_cosmo=True
    )

    oneSample(
        fpl, "all_cosmo", ["av", "bv"],
        args, comm
    )

    oneSample(
        fpl, "no_Ode0", ["Ode0", "av", "bv"],
        args, comm
    )
