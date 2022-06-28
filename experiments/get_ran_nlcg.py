from cmath import exp
from typing import Any, Iterable, Set, Tuple, Union, List, Dict
from pathlib import Path
import json
import multiprocessing as mp
from copy import deepcopy

EXPT_DIR = Path("./expt_rslts/")
OPTIMIZERS = ("fr", "dy", "pr", "hs")


def read_single_expt_specs(expt: Path) -> Dict[str, Any]:
    with open(expt, "r") as jfile:
        full_expt = json.load(jfile)

    specs = full_expt["specs"]
    return specs


def read_existing_expt_specs(expt_dir: Union[Path, str]) -> List[Dict[str, Any]]:
    expt_dir = Path(expt_dir)
    expt_files = expt_dir.glob("*.json")
    with mp.Pool() as pool:
        expt_specs = pool.map(read_single_expt_specs, expt_files)

    return expt_specs


def filter_to_nlcg_expts(all_expts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nlcg = [i for i in all_expts if i["opt"] in OPTIMIZERS]

    return nlcg


def strip_optimizer(expt_spec: Dict[str, Any]) -> Dict[str, Any]:
    expt_spec.pop("opt")
    return expt_spec


def get_specs_per_optimizer(
    expt_list: Iterable, optimizer_name: str
) -> Set[Tuple[Tuple[str, Any], ...]]:
    opt_list = [
        strip_optimizer(deepcopy(i)) for i in expt_list if i["opt"] == optimizer_name
    ]

    opt_set = set([tuple(i.items()) for i in opt_list])

    return opt_set


def get_all_hyperparams(
    expt_list: Iterable[Dict[str, Any]]
) -> Set[Tuple[Tuple[str, Any], ...]]:
    expt_copy = deepcopy(expt_list)
    for expt in expt_copy:
        expt.pop("opt")
    list_of_tuples = [tuple(i.items()) for i in expt_copy]
    set_of_tuples = set(list_of_tuples)

    return set_of_tuples


def re_add_optimizer_and_fix_hparams(hparam_set, name):
    list_of_dicts = [dict(i) for i in hparam_set]
    for dict_ in list_of_dicts:
        dict_["opt"] = name
        dict_["record"] = "./expt_rslts/"
        try:
            if dict_["extrapolation_factor"] > 1.0:
                dict_["extrapolation_factor"] = 1.0 / dict_["extrapolation_factor"]
        except KeyError:
            dict_["extrapolation_factor"] = None

    return list_of_dicts


def get_set_of_extrapolation_factors(nlcg):
    list_of_dicts = [dict(i) for i in nlcg]
    list_of_extraps = []
    for expt in list_of_dicts:
        try:
            extrap = expt["extrapolation_factor"]
        except KeyError:
            extrap = None
        list_of_extraps.append(extrap)

    return set(list_of_extraps)


def main():
    all_expts = read_existing_expt_specs(EXPT_DIR)
    nlcg_expts = filter_to_nlcg_expts(all_expts)
    fr_expts = get_specs_per_optimizer(nlcg_expts, "fr")
    dy_expts = get_specs_per_optimizer(nlcg_expts, "dy")
    hs_expts = get_specs_per_optimizer(nlcg_expts, "hs")
    pr_expts = get_specs_per_optimizer(nlcg_expts, "pr")

    set_of_nlcg_hyper_params = get_all_hyperparams(nlcg_expts)
    print(get_set_of_extrapolation_factors(set_of_nlcg_hyper_params))

    print(len(nlcg_expts))
    print(len(set_of_nlcg_hyper_params))
    print(len(fr_expts | dy_expts | hs_expts | pr_expts))

    # sets are confusing, make sure I'm not doing something dumb
    # does a set of all sets contain itself?
    assert set_of_nlcg_hyper_params == (fr_expts | dy_expts | hs_expts | pr_expts)

    fr_todo = re_add_optimizer_and_fix_hparams(
        set_of_nlcg_hyper_params - fr_expts, "fr"
    )
    pr_todo = re_add_optimizer_and_fix_hparams(
        set_of_nlcg_hyper_params - pr_expts, "pr"
    )
    hs_todo = re_add_optimizer_and_fix_hparams(
        set_of_nlcg_hyper_params - hs_expts, "hs"
    )
    dy_todo = re_add_optimizer_and_fix_hparams(
        set_of_nlcg_hyper_params - dy_expts, "dy"
    )

    master_todo = fr_todo + pr_todo + hs_todo + dy_todo
    print("total todo:", len(master_todo))
    with open("nlcg_todo.json", "w") as jfile:
        json.dump(master_todo, jfile)


if __name__ == "__main__":
    main()
