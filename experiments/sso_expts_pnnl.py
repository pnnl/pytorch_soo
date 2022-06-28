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

import nn

EXPT_SPECS = "expt_specs"
DATASET = "cifar10"

SEEDS = {
    "sgd": [1, 2, 10, 100, 1234, 555, 13, 99, 69, 420],
    "kn": [1, 2, 10, 100],
    "sr1": [1, 2, 10, 100],
    "sr1d": [1, 2, 10, 100],
    "dfp": [1, 2, 10, 100],
    "dfpi": [1, 2, 10, 100],
    "bfgs": [1, 2, 10, 100],
    "bfgsi": [1, 2, 10, 100],
    "fr": [1, 2, 10, 100],
    "pr": [1, 2, 10, 100],
    "hs": [1, 2, 10, 100],
    "dy": [1, 2, 10, 100],
}

# Old ones...
BATCH_SIZES = {
    "sgd": [2, 10, 100, 1000, 10000, 25000],
    "kn": [100, 1000, 10000, 25000, 50000],
    "sr1": [100, 1000, 10000, 25000, 50000],
    "sr1d": [100, 1000, 10000, 25000, 50000],
    "dfp": [100, 1000, 10000, 25000, 50000],
    "dfpi": [100, 1000, 10000, 25000, 50000],
    "bfgs": [100, 1000, 10000, 25000, 50000],
    "bfgsi": [100, 1000, 10000, 25000, 50000],
    "fr": [10],
    "pr": [10],
    "hs": [10],
    "dy": [10],
}
BATCH_SIZES = {
    "sgd": [2, 10, 100, 1000, 10000, 25000],
    "kn": [10, 100, 1000, 5000, 10000, 25000],
    "sr1": [5000],
    "sr1d": [5000],
    "dfp": [5000],
    "dfpi": [5000],
    "bfgs": [5000],
    "bfgsi": [5000],
    "fr": [5000],
    "pr": [5000],
    "hs": [5000],
    "dy": [5000],
}

NUM_EPOCHS = {
    "sgd": [500],
    "kn": [100],
    "sr1": [100],
    "sr1d": [100],
    "dfp": [100],
    "dfpi": [100],
    "bfgs": [100],
    "bfgsi": [100],
    "fr": [100],
    "pr": [100],
    "hs": [100],
    "dy": [100],
}

LEARNING_RATES = {
    "sgd": [0.1, 0.05, 0.01, 0.005, 0.001],
    "kn": [1.0, 0.9, 0.5, 0.1],
    "sr1": [1.0],
    "sr1d": [1.0],
    "dfp": [10.0, 5.0, 1.0, 0.9],
    "dfpi": [10.0, 5.0, 1.0, 0.9],
    "bfgs": [10.0, 5.0, 1.0, 0.9],
    "bfgsi": [10.0, 5.0, 1.0, 0.9],
    "fr": [2.0, 1.0, 0.9, 0.5, 0.1],
    "pr": [2.0, 1.0, 0.9, 0.5, 0.1],
    "hs": [2.0, 1.0, 0.9, 0.5, 0.1],
    "dy": [2.0, 1.0, 0.9, 0.5, 0.1],
}

MOMENTUMS = {
    "sgd": [0.0, 0.1, 0.5, 0.8, 0.9],
    "kn": [0.0],
    "sr1": [0.0],
    "sr1d": [0.0],
    "dfp": [0.0],
    "dfpi": [0.0],
    "bfgs": [0.0],
    "bfgsi": [0.0],
    "fr": [0.0],
    "pr": [0.0],
    "hs": [0.0],
    "dy": [0.0],
}

NEWTON_ITERS = {
    "sgd": [0],
    "kn": [1, 5, 10],
    "sr1": [1, 5, 10],
    "sr1d": [1, 5, 10],
    "dfp": [1, 5, 10],
    "dfpi": [1, 5, 10],
    "bfgs": [1, 5, 10],
    "bfgsi": [1, 5, 10],
    "fr": [1, 10, 50],
    "pr": [1, 10, 50],
    "hs": [1, 10, 50],
    "dy": [1, 10, 50],
}

ABS_NEWTON_TOLS = {
    "sgd": [0],
    "kn": [1e-5],
    "sr1": [1e-1, 1e-4, 1e-5],
    "sr1d": [1e-1, 1e-4, 1e-5],
    "dfp": [1e-1, 1e-4, 1e-5],
    "dfpi": [1e-1, 1e-4, 1e-5],
    "bfgs": [1e-1, 1e-4, 1e-5],
    "bfgsi": [1e-1, 1e-4, 1e-5],
    "fr": [1e-5],
    "pr": [1e-5],
    "hs": [1e-5],
    "dy": [1e-5],
}

REL_NEWTON_TOLS = {
    "sgd": [0],
    "kn": [1e-6],
    "sr1": [1e-6],
    "sr1d": [1e-6],
    "dfp": [1e-6],
    "dfpi": [1e-6],
    "bfgs": [1e-6],
    "bfgsi": [1e-6],
    "fr": [1e-8],
    "pr": [1e-8],
    "hs": [1e-8],
    "dy": [1e-8],
}

CR_ITERS = {
    "sgd": [0],
    "kn": [1, 5, 10],
    "sr1": [0],
    "sr1d": [0],
    "dfp": [1, 10, 50],
    "dfpi": [0],
    "bfgs": [1, 10, 50],
    "bfgsi": [0],
    "fr": [0],
    "pr": [0],
    "hs": [0],
    "dy": [0],
}

CR_TOLS = {
    "sgd": [0],
    "kn": [1e-3],
    "sr1": [0.0],
    "sr1d": [0.0],
    "dfp": [1e-3],
    "dfpi": [0.0],
    "bfgs": [1e-3],
    "bfgsi": [0.0],
    "fr": [0.0],
    "pr": [0.0],
    "hs": [0.0],
    "dy": [0.0],
}

SUFFICIENT_DECREASE = {
    "sgd": [None],
    "kn": [None],
    "sr1": [None],
    "sr1d": [None],
    "dfp": [1e-4, 1e-3, 1e-2],
    "dfpi": [1e-4, 1e-3, 1e-2],
    "bfgs": [1e-4, 1e-3, 1e-2],
    "bfgsi": [1e-4, 1e-3, 1e-2],
    "fr": [1e-3, 1e-4, 1e-5],
    "pr": [1e-3, 1e-4, 1e-5],
    "hs": [1e-3, 1e-4, 1e-5],
    "dy": [1e-3, 1e-4, 1e-5],
}

CURVATURE_CONDITION = {
    "sgd": [None],
    "kn": [None],
    "sr1": [None],
    "sr1d": [None],
    "dfp": [0.99, 0.9, 0.5],
    "dfpi": [0.99, 0.9, 0.5],
    "bfgs": [0.99, 0.9, 0.5],
    "bfgsi": [0.99, 0.9, 0.5],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

EXTRAPOLATION_FACTOR = {
    "sgd": [None],
    "kn": [None],
    "sr1": [0.9, 0.5, 0.1],
    "sr1d": [0.9, 0.5, 0.1],
    "dfp": [0.9, 0.5, 0.1],
    "dfpi": [0.9, 0.5, 0.1],
    "bfgs": [0.9, 0.5, 0.1],
    "bfgsi": [0.9, 0.5, 0.1],
    "fr": [0.9, 0.5, 0.1],
    "pr": [0.9, 0.5, 0.1],
    "hs": [0.9, 0.5, 0.1],
    "dy": [0.9, 0.5, 0.1],
}

MAX_SEARCHES = {
    "sgd": [None],
    "kn": [None],
    "sr1": [2, 5, 10],
    "sr1d": [2, 5, 10],
    "dfp": [2, 5, 10],
    "dfpi": [2, 5, 10],
    "bfgs": [2, 5, 10],
    "bfgsi": [2, 5, 10],
    "fr": [2, 5, 10],
    "pr": [2, 5, 10],
    "hs": [2, 5, 10],
    "dy": [2, 5, 10],
}

QUASI_NEWTON_MEMORY = {
    "sgd": [None],
    "kn": [0],
    "sr1": [1, 2, 5, 10],
    "sr1d": [1, 2, 5, 10],
    "dfp": [1, 2, 5, 10],
    "dfpi": [1, 2, 5, 10],
    "bfgs": [1, 2, 5, 10],
    "bfgsi": [1, 2, 5, 10],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

INITIAL_RADIUS = {
    "sgd": [None],
    "kn": [None],
    "sr1": [0.1, 1.0, 10.0],
    "sr1d": [0.1, 1.0, 10.0],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

NABLA0 = {
    "sgd": [None],
    "kn": [None],
    "sr1": [0.0, 1e-4],
    "sr1d": [0.0, 1e-4],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

NABLA1 = {
    "sgd": [None],
    "kn": [None],
    "sr1": [0.1, 0.25],
    "sr1d": [0.1, 0.25],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

NABLA2 = {
    "sgd": [None],
    "kn": [None],
    "sr1": [0.5, 0.75, 0.9],
    "sr1d": [0.5, 0.75, 0.9],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

SHRINK_FACTOR = {
    "sgd": [None],
    "kn": [None],
    "sr1": [0.1, 0.25, 0.5],
    "sr1d": [0.1, 0.25, 0.5],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

GROWTH_FACTOR = {
    "sgd": [None],
    "kn": [None],
    "sr1": [2.0, 4.0, 10.0],
    "sr1d": [2.0, 4.0, 10.0],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}

MAX_SUBPROBLEM_ITER = {
    "sgd": [None],
    "kn": [None],
    "sr1": [5, 10, 25],
    "sr1d": [5, 10, 25],
    "dfp": [None],
    "dfpi": [None],
    "bfgs": [None],
    "bfgsi": [None],
    "fr": [None],
    "pr": [None],
    "hs": [None],
    "dy": [None],
}


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


def spec_to_namespace(spec, opt, outdir):
    batch_size_train = spec[0]
    num_epoch = spec[1]
    learning_rate = spec[2]
    momentum = spec[3]
    newton_iter = spec[4]
    abs_newton_tol = spec[5]
    rel_newton_tol = spec[6]
    cr_iter = spec[7]
    cr_tol = spec[8]
    try:
        sufficient_decrease = float(spec[9])
    except TypeError:
        sufficient_decrease = None
    try:
        curvature_condition = float(spec[10])
    except TypeError:
        curvature_condition = None
    try:
        extrapolation_factor = float(spec[11])
    except TypeError:
        extrapolation_factor = None
    try:
        max_searches = int(spec[12])
    except TypeError:
        max_searches = None

    if None in (sufficient_decrease, extrapolation_factor, max_searches):
        sufficient_decrease = (
            curvature_condition
        ) = extrapolation_factor = max_searches = None

    if batch_size_train == 50000:
        newton_iter = 1

    try:
        memory = int(spec[13])
    except TypeError:
        memory = None
    initial_radius = spec[14]
    nabla0 = spec[15]
    nabla1 = spec[16]
    nabla2 = spec[17]
    shrink_factor = spec[18]
    growth_factor = spec[19]
    max_subproblem_iter = spec[20]
    seed = spec[21]

    expt_spec = SimpleNamespace(
        opt=opt,
        dataset=DATASET,
        batch_size_train=batch_size_train,
        batch_size_test=10000,
        momentum=momentum,
        hidden=15,
        max_newton=newton_iter,
        abs_newton_tol=abs_newton_tol,
        rel_newton_tol=rel_newton_tol,
        max_cr=cr_iter,
        cr_tol=cr_tol,
        learning_rate=learning_rate,
        sufficient_decrease=sufficient_decrease,
        curvature_condition=curvature_condition,
        extrapolation_factor=extrapolation_factor,
        max_searches=max_searches,
        num_epoch=num_epoch,
        seed=seed,
        read_nn=None,
        write_nn=True,
        log_interval=10,
        device="cuda",
        record=outdir,
        memory=memory,
        initial_radius=initial_radius,
        nabla0=nabla0,
        nabla1=nabla1,
        nabla2=nabla2,
        shrink_factor=shrink_factor,
        growth_factor=growth_factor,
        max_subproblem_iter=max_subproblem_iter,
    )

    return expt_spec


def generate_specs(proc_id, total_procs, opt, outdir, force):
    specfile = f"{EXPT_SPECS}_{opt}.pkl"
    generate = proc_id in (0, -1) and (not os.path.exists(specfile) or force)
    if generate:
        batch_sizes_train = BATCH_SIZES[opt]
        num_epochs = NUM_EPOCHS[opt]
        learning_rates = LEARNING_RATES[opt]
        momentums = MOMENTUMS[opt]
        newton_iters = NEWTON_ITERS[opt]
        abs_newton_tols = ABS_NEWTON_TOLS[opt]
        rel_newton_tols = REL_NEWTON_TOLS[opt]
        cr_iters = CR_ITERS[opt]
        cr_tols = CR_TOLS[opt]
        sufficient_decreases = SUFFICIENT_DECREASE[opt]
        curvature_conditions = CURVATURE_CONDITION[opt]
        extrapolation_factors = EXTRAPOLATION_FACTOR[opt]
        max_searches = MAX_SEARCHES[opt]
        memory = QUASI_NEWTON_MEMORY[opt]
        initial_radius = INITIAL_RADIUS[opt]
        nabla0 = NABLA0[opt]
        nabla1 = NABLA1[opt]
        nabla2 = NABLA2[opt]
        shrink_factor = SHRINK_FACTOR[opt]
        growth_factor = GROWTH_FACTOR[opt]
        max_subproblem_iter = MAX_SUBPROBLEM_ITER[opt]
        seeds = SEEDS[opt]

        expt_specs = itertools.product(
            batch_sizes_train,  # 0
            num_epochs,  # 1
            learning_rates,  # 2
            momentums,  # 3
            newton_iters,  # 4
            abs_newton_tols,  # 5
            rel_newton_tols,  # 6
            cr_iters,  # 7
            cr_tols,  # 8
            sufficient_decreases,  # 9
            curvature_conditions,  # 10
            extrapolation_factors,  # 11
            max_searches,  # 12
            memory,  # 13
            initial_radius,  # 14
            nabla0,  # 15
            nabla1,  # 16
            nabla2,  # 17
            shrink_factor,  # 18
            growth_factor,  # 19
            max_subproblem_iter,  # 20
            seeds,  # 21
        )

        print(expt_specs)

        expt_specs = [*expt_specs]

        expt_specs = [spec_to_namespace(i, opt, outdir) for i in expt_specs]

        with open(specfile, "wb") as pkl:
            pickle.dump(expt_specs, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Reading expt pickle...")
        with open(specfile, "rb") as pkl:
            expt_specs = pickle.load(pkl)

    if -1 == proc_id:
        sublist = expt_specs
    else:
        sublist = [expt_specs[i::total_procs] for i in range(total_procs)][proc_id]

    print("Total expts: {}, to be ran: {}".format(len(expt_specs), len(sublist)))

    return sublist


def main():
    proc_id, total, opt, outdir, force = get_args()
    expt_list_ = generate_specs(proc_id, total, opt, outdir, force)
    if force:
        return
    random.shuffle(expt_list_)
    # make a subset
    expt_list = expt_list_[:100]
    for expt in expt_list:
        print("Running:", expt)
        try:
            nn.main(**vars(expt))
        except Exception as e:
            print("=" * 80)
            traceback.print_exc()
            print("=" * 80)


if __name__ == "__main__":
    main()
