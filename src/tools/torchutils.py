import torch
from torch.nn import functional as F


class Accuracy:
    def __call__(self, y_pred, y):
        res = torch.argmax(input=y_pred, dim=1)
        res = (res == y).type(torch.FloatTensor)

        accuracy = torch.mean(input=res)

        return accuracy.item()


class SoftCrossEntropyLoss:
    def __init__(self, weights):
        super(SoftCrossEntropyLoss).__init__()
        self.weights = weights

    def __call__(self, y_hat, y):
        p = F.log_softmax(input=y_hat, dim=1)
        w_labels = self.weights * y
        loss = - (w_labels * p).sum() / w_labels.sum()
        return loss


def get_class_weights(answers_frequency, idx_to_answer):
    weights = torch.zeros(size=(len(idx_to_answer), ), dtype=torch.float)
    minimum = min(answers_frequency.items(), key=lambda x: x[1])[1]
    maximum = max(answers_frequency.items(), key=lambda x: x[1])[1]

    for i in range(len(weights)):
        weights[i] = maximum / answers_frequency[idx_to_answer[i]]

    weights = torch.log(input=weights)

    return weights
