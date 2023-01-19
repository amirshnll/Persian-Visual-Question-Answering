import os


def get_VQAv1_configs(root_path):

    cfgs = {}

    # ---------------- Original Dataset Configurations ----------------

    # Train Data Configuration
    cfgs['train'] = {}
    cfgs['train']['annotations'] = {
        'file_name': 'mscoco_train2014_annotations.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip',
        'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    }
    # cfgs['train']['captions'] = {
    #     'file_name': 'captions_train2014.json',
    #     'link': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    # }
    cfgs['train']['images'] = {
        'file_name': 'train2014',
        'link': 'http://images.cocodataset.org/zips/train2014.zip',
        'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    }
    # cfgs['train']['instances'] = {
    #     'file_name': 'instances_train2014.json',
    #     'link': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    # }
    # cfgs['train']['person_keypoints'] = {
    #     'file_name': 'person_keypoints_train2014.json',
    #     'link': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    # }
    # cfgs['train']['multiple_choice_questions'] = {
    #     'file_name': 'MultipleChoice_mscoco_train2014_questions.json',
    #     'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    # }
    cfgs['train']['open_ended_questions'] = {
        'file_name': 'OpenEnded_mscoco_train2014_questions.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip',
        'path': os.path.join(root_path, 'original', 'VQAv1', 'train')
    }

    # Val Data Configuration
    cfgs['val'] = {}
    cfgs['val']['annotations'] = {
        'file_name': 'mscoco_val2014_annotations.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip',
        'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    }
    # cfgs['val']['captions'] = {
    #     'file_name': 'captions_val2014.json',
    #     'link': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    # }
    cfgs['val']['images'] = {
        'file_name': 'val2014',
        'link': 'http://images.cocodataset.org/zips/val2014.zip',
        'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    }
    # cfgs['val']['instances'] = {
    #     'file_name': 'instances_val2014.json',
    #     'link': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    # }
    # cfgs['val']['person_keypoints'] = {
    #     'file_name': 'person_keypoints_val2014.json',
    #     'link': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    # }
    # cfgs['val']['multiple_choice_questions'] = {
    #     'file_name': 'MultipleChoice_mscoco_val2014_questions.json',
    #     'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    # }
    cfgs['val']['open_ended_questions'] = {
        'file_name': 'OpenEnded_mscoco_val2014_questions.json',
        'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip',
        'path': os.path.join(root_path, 'original', 'VQAv1', 'val')
    }

    # Test Data Configuration
    # cfgs['test'] = {}
    # cfgs['test']['images'] = {
    #     'file_name': 'test2015',
    #     'link': 'http://images.cocodataset.org/zips/test2015.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'test')
    # }
    # cfgs['test']['image_info'] = {
    #     'file_name': 'image_info_test2015.json',
    #     'link': 'http://images.cocodataset.org/annotations/image_info_test2015.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'test')
    # }
    # cfgs['test']['multiple_choice_questions'] = {
    #     'file_name': 'MultipleChoice_mscoco_test2015_questions.json',
    #     'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'test')
    # }
    # cfgs['test']['open_ended_questions'] = {
    #     'file_name': 'OpenEnded_mscoco_test2015_questions.json',
    #     'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'test')
    # }

    # Dev Data Configurations
    # cfgs['dev'] = {}
    # cfgs['dev']['images'] = {
    #     'file_name': 'test2015',
    #     'link': 'http://images.cocodataset.org/zips/test2015.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'test')
    # }
    # cfgs['dev']['image_info'] = {
    #     'file_name': 'image_info_test-dev2015.json',
    #     'link': 'http://images.cocodataset.org/annotations/image_info_test2015.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'dev')
    # }
    # cfgs['dev']['multiple_choice_questions'] = {
    #     'file_name': 'MultipleChoice_mscoco_test-dev2015_questions.json',
    #     'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'dev')
    # }
    # cfgs['dev']['open_ended_questions'] = {
    #     'file_name': 'OpenEnded_mscoco_test-dev2015_questions.json',
    #     'link': 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip',
    #     'path': os.path.join(root_path, 'original', 'VQAv1', 'dev')
    # }
    # ---------------------------------------------------------------------

    return cfgs

