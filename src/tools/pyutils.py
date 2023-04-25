import os
import json
import torch
import random
import numpy as np
import pickle as pk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_file(data, path, file_name, file_type):
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    if file_type == 'json':
        with open(file=os.path.join(path, file_name), mode='w') as fp:
            json.dump(obj=data, fp=fp, indent=4)
    else:
        with open(file=os.path.join(path, file_name), mode='wb') as fp:
            pk.dump(obj=data, file=fp, protocol=pk.HIGHEST_PROTOCOL)


def load_file(path):  # sourcery skip: raise-specific-error
    extension = path.split(sep='.')[-1]

    if extension == 'json':
        with open(file=path, mode='r') as fp:
            data = json.load(fp=fp)

    elif extension in ['pickle', 'pk']:
        with open(file=path, mode='rb') as fp:
            data = pk.load(file=fp)
    else:
        raise Exception('Unknown file format!')

    return data
