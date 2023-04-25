import gc
import torch
import numpy as np
from tqdm import tqdm
from src.tools.torchutils import Accuracy


def val(model, criterion, dataloader, device, global_step):
    print(f'{"-" * 10} Validating {"-" * 10}')

    model.eval()
    t = tqdm(dataloader)
    accuracy_fn = Accuracy()

    scores = {
        'loss': [],
        'accuracy': []
    }

    for image, features, questions, answers in t:
        features = features.to(device=device)
        questions = questions.to(device=device)
        answers = answers.to(device=device)

        with torch.no_grad():
            preds = model(features, questions)

        loss_score = criterion(input=preds, target=answers)

        # Move labels, preds, and loss to cpu
        preds = preds.cpu().detach()
        answers = answers.cpu().detach()

        acc_score = accuracy_fn(y_pred=preds, y=answers)
        loss_score = loss_score.cpu().detach().item()

        scores['loss'].append(loss_score)
        scores['accuracy'].append(acc_score)

        global_step += 1
        t.set_description(desc=f"Val Loss: {loss_score:0.4f} Accuracy: {acc_score:0.4f}")

    loss_mean = np.mean(a=scores['loss'])
    acc_mean = np.mean(a=scores['accuracy'])
    print(f'Val Loss: {loss_mean}, Accuracy: {acc_mean}')

    scores = {
        'loss': loss_mean,
        'accuracy': acc_mean
    }

    gc.collect()
    torch.cuda.empty_cache()

    return scores, global_step
