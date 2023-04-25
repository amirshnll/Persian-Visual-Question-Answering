import os
import cv2
import torch
from tqdm import tqdm
from torchvision import transforms as T

from src.models.resnexts import FeatureExtractor
from src.tools.pyutils import save_file, load_file


def UnNormalize(images):
    trans = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    return trans(images)


def extract_features(data, trans, device, save_path):

    FE = FeatureExtractor()
    FE = FE.to(device=device)
    FE.eval()

    for i, d in enumerate(tqdm(data)):
        ip = d['image_path']

        save_name = f"{ip.split(sep='/')[-1].split(sep='.')[0]}.pickle"

        if not os.path.exists(path=os.path.join(save_path, save_name)):
            img = cv2.imread(filename=ip)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
            img = trans(img).unsqueeze(0)
            img = img.to(device=device)

            with torch.no_grad():
                features = FE(img).cpu().detach().numpy()

            save_file(data=features, path=save_path, file_name=save_name, file_type='pickle')

        # features = load_file(path=os.path.join(save_path, save_name))
        fp = os.path.join(save_path, save_name)
        d['feature_path'] = fp

    del FE

    return data

