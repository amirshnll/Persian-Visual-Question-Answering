import os
import json

import cv2
import torch
import shutil
from zipfile import ZipFile
from urllib.request import urlretrieve
from src.tools.pyutils import load_file
from torch.utils.data import Dataset, DataLoader


def check_VQAv1_dataset(dataset_paths):
    for key, value in dataset_paths.items():
        for k, v in value.items():
            link = v['link']
            path = v['path']
            file_name = v['file_name']
            zip_name = link.split('/')[-1]

            if not os.path.exists(os.path.join(path, file_name)):
                print(f'path not found: {os.path.join(path, file_name)}')
                print(f'searching for download file...')

                if not os.path.exists(os.path.join(path, zip_name)):
                    print(f'download file not found: {os.path.join(path, zip_name)}')
                    print('downloading...')
                    os.makedirs(name=path, exist_ok=True)
                    urlretrieve(url=link, filename=os.path.join(path, zip_name))

                print(f'download file found: {os.path.join(path, zip_name)}')
                print('extracting...')
                with ZipFile(file=os.path.join(path, zip_name), mode='r') as ref:
                    ref.extractall(path=path)

                if zip_name == 'annotations_trainval2014.zip':
                    os.makedirs(name=dataset_paths['val']['captions']['path'], exist_ok=True)
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['train']['captions']['file_name']), dst=os.path.join(dataset_paths['train']['captions']['path'], dataset_paths['train']['captions']['file_name']))
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['val']['captions']['file_name']), dst=os.path.join(dataset_paths['val']['captions']['path'], dataset_paths['val']['captions']['file_name']))

                    os.makedirs(name=dataset_paths['val']['instances']['path'], exist_ok=True)
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['train']['instances']['file_name']), dst=os.path.join(dataset_paths['train']['instances']['path'], dataset_paths['train']['instances']['file_name']))
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['val']['instances']['file_name']), dst=os.path.join(dataset_paths['val']['instances']['path'], dataset_paths['val']['instances']['file_name']))

                    os.makedirs(name=dataset_paths['val']['person_keypoints']['path'], exist_ok=True)
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['train']['person_keypoints']['file_name']), dst=os.path.join(dataset_paths['train']['person_keypoints']['path'], dataset_paths['train']['person_keypoints']['file_name']))
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['val']['person_keypoints']['file_name']), dst=os.path.join(dataset_paths['val']['person_keypoints']['path'], dataset_paths['val']['person_keypoints']['file_name']))
                    os.rmdir(path=os.path.join(path, 'annotations'))

                elif zip_name == 'image_info_test2015.zip':
                    os.makedirs(name=dataset_paths['dev']['image_info']['path'], exist_ok=True)
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['test']['image_info']['file_name']), dst=os.path.join(dataset_paths['test']['image_info']['path'], dataset_paths['test']['image_info']['file_name']))
                    shutil.move(src=os.path.join(path, 'annotations', dataset_paths['dev']['image_info']['file_name']), dst=os.path.join(dataset_paths['dev']['image_info']['path'], dataset_paths['dev']['image_info']['file_name']))
                    os.rmdir(path=os.path.join(path, 'annotations'))

                elif zip_name == 'Questions_Test_mscoco.zip':
                    os.makedirs(name=dataset_paths['dev']['multiple_choice_questions']['path'], exist_ok=True)
                    shutil.move(src=os.path.join(path, dataset_paths['dev']['multiple_choice_questions']['file_name']), dst=os.path.join(dataset_paths['dev']['multiple_choice_questions']['path'], dataset_paths['dev']['multiple_choice_questions']['file_name']))

                    os.makedirs(name=dataset_paths['dev']['open_ended_questions']['path'], exist_ok=True)
                    shutil.move(src=os.path.join(path, dataset_paths['dev']['open_ended_questions']['file_name']), dst=os.path.join(dataset_paths['dev']['open_ended_questions']['path'], dataset_paths['dev']['open_ended_questions']['file_name']))

                print('extracted. removing download file...')
                os.remove(path=os.path.join(path, zip_name))
                print('done.')


def get_VQAv1_dataset(dataset_paths):
    data = {}

    for key, value in dataset_paths.items():
        print(f'preparing {key} data ...')
        data[key] = []
        image_path = os.path.join(dataset_paths[key]['images']['path'], dataset_paths[key]['images']['file_name'])

        annotations, a_subtype = get_annotations(
            file_path=dataset_paths[key]['annotations']['path'],
            file_name=dataset_paths[key]['annotations']['file_name']
        )
        questions, q_subtype = get_questions(
            file_path=dataset_paths[key]['open_ended_questions']['path'],
            file_name=dataset_paths[key]['open_ended_questions']['file_name']
        )

        uqqids = [k for k, v in questions.items()]
        uaqids = [k for k, v in annotations.items()]
        assert a_subtype == q_subtype, 'annotation subtype != question subtype'
        assert len(annotations) == len(questions), 'number of annotations != number of questions'
        assert len(set(uaqids + uqqids)) == len(annotations) == len(questions), 'unique QA conflict'

        for aqid, a in annotations.items():
            d = {}
            q = questions[aqid]

            assert a['image_id'] == q['image_id'], 'image_id conflict'

            d.update(a)
            d.update(q)
            d['image_id_str'] = '{0:012d}'.format(int(d['image_id']))
            d['image_path'] = os.path.join(
                image_path, "COCO_{0}_{1:012d}.jpg".format(a_subtype, int(d['image_id']))
            )

            data[key].append(d)

    return data


def get_annotations(file_path, file_name):

    annotations = {}
    data = json.load(fp=open(file=os.path.join(file_path, file_name), mode='r'))
    data_subtype = data['data_subtype']
    data = data['annotations']

    for d in data:
        qid = d['question_id']
        annotations[qid] = {
            'en_answer': d['multiple_choice_answer'],
            'image_id': d['image_id'],
            'question_id': qid,
        }

    return annotations, data_subtype


def get_questions(file_path, file_name):
    questions = {}
    data = json.load(fp=open(file=os.path.join(file_path, file_name), mode='r'))
    data_subtype = data['data_subtype']
    data = data['questions']

    for d in data:
        qid = d['question_id']
        questions[qid] = {
            'en_question': d['question'],
            'image_id': d['image_id'],
            'question_id': qid,
        }

    return questions, data_subtype


class VQAv1Dataset(Dataset):
    def __init__(self, data, token_to_idx, answer_to_idx, tokens_length, trans):
        super(VQAv1Dataset, self).__init__()
        self.data = data
        self.token_to_idx = token_to_idx
        self.answer_to_idx = answer_to_idx
        self.tokens_length = tokens_length
        self.trans = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        q = d['tokens']
        a = d['fa_answer']
        ip = d['image_path']
        fp = d['feature_path']
        question = torch.zeros(size=(self.tokens_length, ), dtype=torch.long)
        for i in range(len(q)):
            if i < self.tokens_length:
                question[i] = self.token_to_idx.get(q[i], self.token_to_idx['UKN'])

        answer = torch.tensor(data=self.answer_to_idx[a], dtype=torch.long)

        image = cv2.imread(filename=ip)
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        image = self.trans(image)

        features = torch.tensor(data=load_file(path=fp), dtype=torch.float).squeeze(0)

        return image, features, question, answer


def get_data_loader(data, token_to_idx, answer_to_idx, tokens_length, batch_size, shuffle, trans):

    dataset = VQAv1Dataset(
        data=data,
        token_to_idx=token_to_idx,
        answer_to_idx=answer_to_idx,
        tokens_length=tokens_length,
        trans=trans
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader
