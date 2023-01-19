import os
from src.tools.pyutils import load_file, save_file
from src.datasets.VQAv1 import check_VQAv1_dataset, get_VQAv1_dataset


def get_original_data(cfgs):
    pp_cfgs = cfgs['preprocess']
    original_data = {}

    for dataset in cfgs['datasets']:
        original_data[dataset] = {}
        save_path = os.path.join(pp_cfgs['root_path'], pp_cfgs[f'{dataset}_original_data'])

        if os.path.exists(save_path):
            print(f'Found {dataset} original data')

            data = load_file(path=save_path)
            original_data[dataset].update(data)
        else:
            print(f'Getting {dataset} original data from scratch...')
            if dataset == 'VQAv1':
                check_VQAv1_dataset(dataset_paths=cfgs['datasets'][dataset])
                data = get_VQAv1_dataset(dataset_paths=cfgs['datasets'][dataset])

                save_file(
                    data=data,
                    path=pp_cfgs['root_path'],
                    file_name=pp_cfgs[f'{dataset}_original_data'],
                    file_type='pickle'
                )

                original_data[dataset].update(data)

    return original_data

