import os
from configurations.datasets.VQAv1 import get_VQAv1_configs


def get_configs():

    cfgs = {
        'session': 'final',
        'seed': 42,
        'root_path': os.getcwd(),
        'epochs': 50,
        'batch_size': 750,
        'lr': 0.0001,
        'datasets': {},
    }

    ### VQAv1:
    cfgs['datasets']['VQAv1'] = get_VQAv1_configs(root_path=os.path.join(cfgs['root_path'], 'datasets'))
    # ---------------- End of original dataset configurations

    # ---------------- Preprocess Configurations ----------------
    cfgs['preprocess'] = {}
    cfgs['preprocess']['root_path'] = os.path.join(cfgs['root_path'], 'datasets', 'preprocessed')
    cfgs['preprocess']['VQAv1_original_data'] = 'VQAv1_original_data.pickle'
    cfgs['preprocess']['dictionary_name'] = 'dictionary.pickle'
    cfgs['preprocess']['idx_to_token'] = 'idx_to_token.pickle'
    cfgs['preprocess']['token_to_idx'] = 'token_to_idx.pickle'
    cfgs['preprocess']['idx_to_answer'] = 'idx_to_answer.pickle'
    cfgs['preprocess']['answer_to_idx'] = 'answer_to_idx.pickle'
    cfgs['preprocess']['image_features'] = os.path.join(cfgs['preprocess']['root_path'], 'image_features')
    # ---------------- End of Preprocess Configurations ----------------

    # Checkpoints Configurations
    cfgs['checkpoints_path'] = os.path.join(cfgs['root_path'], 'checkpoints', cfgs['session'])
    cfgs['logs_path'] = os.path.join(cfgs['checkpoints_path'], 'logs')

    # Create Necessary Directories
    os.makedirs(name=cfgs['checkpoints_path'], exist_ok=True)
    os.makedirs(name=cfgs['logs_path'], exist_ok=True)

    return cfgs



