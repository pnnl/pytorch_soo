import json
import pickle
from types import SimpleNamespace

if __name__ == "__main__":
    with open("nlcg_todo.json", "r") as jfile:
        todo = json.load(jfile)

    namespaces = [SimpleNamespace(**i) for i in todo]
    with open("expt_specs.pkl", "wb") as pkl_file:
        pickle.dump(namespaces, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
