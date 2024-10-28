import os
import json
import random
import numpy as np
import torch

# from datasets import CustomImageFolder


def write_dictionary(dictionary_to_wrtie, file_name="results.json"):
    stored_json = json.dumps(dictionary_to_wrtie)
    with open(file_name, "w") as file:
        file.write(stored_json)


def load_dictionary(file_name="results.json"):
    if os.path.isfile(file_name) and os.access(file_name, os.R_OK):
        with open(file_name) as json_file:
            loaded_dictionary = json.load(json_file)
        return loaded_dictionary
    else:
        print("File does not exist, creating a new one")
        with open(file_name, "w") as json_file:
            json_file.write(json.dumps({}))
        return load_dictionary(file_name)


def initialize_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
