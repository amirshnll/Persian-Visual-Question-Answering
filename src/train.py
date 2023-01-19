import gc
import torch
import numpy as np
from tqdm import tqdm

from src.tools.torchutils import Accuracy


def train(model, optimizer, criterion, dataloader, device, global_step):
    print(f'{"-" * 10} Training {"-" * 10}')

    model.train()
    t = tqdm(dataloader)
    accuracy_fn = Accuracy()

    scores = {
        'loss': [],
        'accuracy': []
    }

    for j, (images, features, questions, answers) in enumerate(t):

        features = features.to(device=device)
        questions = questions.to(device=device)
        answers = answers.to(device=device)

        preds = model(features, questions)

        loss_score = criterion(input=preds, target=answers)

        optimizer.zero_grad()
        loss_score.backward()
        optimizer.step()

        # Move answers, preds, and loss to cpu
        preds = preds.cpu().detach()
        answers = answers.cpu().detach()
        loss_score = loss_score.cpu().detach().item()
        acc_score = accuracy_fn(y_pred=preds, y=answers)

        scores['loss'].append(loss_score)
        scores['accuracy'].append(acc_score)

        global_step += 1
        t.set_description(desc=f"Train Loss: {loss_score:0.4f} Accuracy: {acc_score:0.4f}")

    print(f'{"-" * 10} End of Training {"-" * 10}')

    loss_mean = np.mean(a=scores['loss'])
    acc_mean = np.mean(a=scores['accuracy'])
    print(f'Train Loss: {loss_mean}, Accuracy: {acc_mean}')

    scores = {
        'loss': loss_mean,
        'accuracy': acc_mean
    }

    gc.collect()
    torch.cuda.empty_cache()

    return model, scores, global_step
