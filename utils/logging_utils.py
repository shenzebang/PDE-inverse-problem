from typing import Dict
import jax.numpy as jnp
import pandas as pd
import os, json

def save_to_csv(data_dict: Dict, save_file: str):
    data_arrays = jnp.stack([jnp.array(data_dict[key]) for key in data_dict], axis=1)
    result = pd.DataFrame(data_arrays, columns=list(data_dict.keys()))
    result.to_csv(save_file, index=False)


def save_config(save_directory, args):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with open(save_directory + '/config.json', 'w') as f:
        json.dump(vars(args), f)