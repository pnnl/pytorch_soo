"""
A testbench for running neural network tests with Second Order Optimizers
Copyright 2021 Nanmiao Wu, Eric Silk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import random
import pickle
import argparse
import traceback
import itertools
from types import SimpleNamespace

import nn_levenberg

EXPT_SPECS = "expt_specs_levenberg.pkl"
DATASET = "cifar10"

SEEDS = {
    "levenberg": [1, 2, 10, 100, 1234, 555, 13, 99],
}

BATCH_SIZES = {
    "levenberg": [100, 1000, 10000, 25000, 50000],
}

NUM_EPOCHS = {
    "levenberg": [100],
}

LEARNING_RATES = {
    "levenberg": [1.0, 0.9, 0.5, 0.1],
}

NEWTON_ITERS = {
    "levenberg": [1, 10, 20],
}

NEWTON_TOLS = {
    "levenberg": [1e-4, 1e-5, 1e-6],
}

CR_ITERS = {
    "levenberg": [1, 2, 10, 20],
}

CR_TOLS = {
    "levenberg": [1e-2, 1e-3, 1e-4],
}

LAMBDA0 = {"levenberg": [0.1, 1.0, 10.0]}

MAX_LAMBDA = {"levenberg": [1e3, 1e2, 1e1]}

MIN_LAMBDA = {"levenberg": [1e-3, 1e-4, 1e-5]}

NU = {"levenberg": [1.1, 2.0**0.5, 2, 10]}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_id",
        type=int,
        default=-1,
        help="The slurm job array task ID. (default: -1)",
    )
    parser.add_argument(
        "--total_tasks",
        type=int,
        default=-1,
        help="The total number of jobs in the array (default: 1)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="The optimizer to set up expts for"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./expt_rslts/",
        help="The directory to store results in (default: ./expt_rslts/)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recreation of the expt pkl and exit"
    )

    args = parser.parse_args()

    return (args.task_id, args.total_tasks, args.optimizer, args.outdir, args.force)


def generate_specs(proc_id, total_procs, opt, outdir, force):
    generate = proc_id in (0, -1) and (not os.path.exists(EXPT_SPECS) or force)
    if generate:
        batch_sizes_train = BATCH_SIZES[opt]
        num_epochs = NUM_EPOCHS[opt]
        learning_rates = LEARNING_RATES[opt]
        newton_iters = NEWTON_ITERS[opt]
        newton_tols = NEWTON_TOLS[opt]
        cr_iters = CR_ITERS[opt]
        cr_tols = CR_TOLS[opt]
        seeds = SEEDS[opt]
        lambda0 = LAMBDA0[opt]
        max_lambda = MAX_LAMBDA[opt]
        min_lambda = MIN_LAMBDA[opt]
        nu = NU[opt]

        expt_specs = itertools.product(
            batch_sizes_train,
            num_epochs,
            learning_rates,
            newton_iters,
            newton_tols,
            cr_iters,
            cr_tols,
            seeds,
            lambda0,
            max_lambda,
            min_lambda,
            nu,
        )

        print(expt_specs)

        expt_specs = [*expt_specs]

        expt_specs = [spec_to_namespace(i, opt, outdir) for i in expt_specs]

        with open(EXPT_SPECS, "wb") as pkl:
            pickle.dump(expt_specs, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Reading expt pickle...")
        with open(EXPT_SPECS, "rb") as pkl:
            expt_specs = pickle.load(pkl)

    if -1 == proc_id:
        sublist = expt_specs
    else:
        sublist = [expt_specs[i::total_procs] for i in range(total_procs)][proc_id]

    print("Total expts: {}, to be ran: {}".format(len(expt_specs), len(sublist)))

    return sublist


def spec_to_namespace(spec, opt, outdir):
    batch_size_train = spec[0]
    num_epoch = spec[1]
    learning_rate = spec[2]
    newton_iter = spec[3]
    newton_tol = spec[4]
    cr_iter = spec[5]
    cr_tol = spec[6]
    seed = spec[7]
    lambda0 = spec[8]
    max_lambda = spec[9]
    min_lambda = spec[10]
    nu = spec[11]

    expt_spec = SimpleNamespace(
        opt=opt,
        dataset=DATASET,
        batch_size_train=batch_size_train,
        batch_size_test=10000,
        hidden=15,
        max_newton=newton_iter,
        newton_tol=newton_tol,
        max_cr=cr_iter,
        cr_tol=cr_tol,
        learning_rate=learning_rate,
        lambda0=lambda0,
        max_lambda=max_lambda,
        min_lambda=min_lambda,
        nu=nu,
        num_epoch=num_epoch,
        seed=seed,
        read_nn=None,
        write_nn=None,
        log_interval=10,
        device="cuda",
        record=outdir,
    )

    return expt_spec


def main():
    proc_id, total, opt, outdir, force = get_args()
    expt_list = generate_specs(proc_id, total, opt, outdir, force)
    if force:
        return
    random.shuffle(expt_list)
    for expt in expt_list:
        print("Running:", expt)
        try:
            nn_levenberg.main(**vars(expt))
        except Exception as e:
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)


if __name__ == "__main__":
    main()
