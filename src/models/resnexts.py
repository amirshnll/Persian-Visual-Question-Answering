import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnext101_32x8d


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        model = resnext101_32x8d(pretrained=True)
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ImageModel(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ImageModel, self).__init__()

        self.transformation = nn.Linear(in_features=2048, out_features=hidden_size)

    def forward(self, x):  # (N, 2048, 7, 7)

        x = x.view(x.size(0), x.size(1), -1)  # (N, 2048, 49)
        x = x.transpose(dim0=2, dim1=1)  # (N, 49, 2048)
        x = self.transformation(x)  # (N, 49, hidden_size)
        x = F.normalize(input=x, p=2, dim=-1)  # (N, 49, hidden_size)

        return x  # (N, 49, hidden_size)


class QuestionModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, embed_dim, dropout, tokens_length):
        super(QuestionModel, self).__init__()

        # Word Processing
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embed_dim, padding_idx=0)
        self.WSA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout, batch_first=True)

        # Sentence Processing
        self.LSTM = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=False,
            dropout=dropout,
            batch_first=True
        )

        nn.init.xavier_uniform_(tensor=self.embedding.weight)

    def forward(self, x, mask):  # (N, T)

        # Word Processing:
        word = self.embedding(x)  # (N, T, hidden_size)
        word, weight = self.WSA(word, word, word, key_padding_mask=mask)  # (N, T, hidden_size), (N, T, T)

        # Question Processing
        question, (hn, cn) = self.LSTM(word)  # (N, T, hidden_size)
        question = F.normalize(input=question, p=2, dim=-1)

        return word, question  # (N, T, hidden_size)


class Model(nn.Module):
    def __init__(self, vocabulary_size, num_classes, tokens_length, hidden_size=512, embed_dim=512, dropout=0.3):
        super(Model, self).__init__()

        self.image_model = ImageModel(hidden_size=hidden_size, dropout=dropout)
        self.question_model = QuestionModel(
            vocabulary_size=vocabulary_size,
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            dropout=dropout,
            tokens_length=tokens_length,
        )

        self.WIMHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
            kdim=embed_dim,
            vdim=embed_dim
        )
        self.SIMHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
            kdim=hidden_size,
            vdim=hidden_size
        )

        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features=49 * hidden_size, out_features=8 * hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(in_features=8 * hidden_size, out_features=num_classes)

    def forward(self, img_features, questions):  # (N, 2048, 7, 7), (N, T)

        mask = (questions == 0)

        # Image Processing
        img_fea = self.image_model(img_features)  # (N, 49, hidden_size)

        # Text Processing
        word_fea, sentence_fea = self.question_model(questions, mask)  # (N, T, embed_dim), (N, T, hidden_size)

        WI_out, _ = self.WIMHA(img_fea, word_fea, word_fea, key_padding_mask=mask)  # (N, 49, hidden_size)
        SI_out, _ = self.SIMHA(img_fea, sentence_fea, sentence_fea, key_padding_mask=mask)  # (N, 49, hidden_size)

        out = torch.flatten(input=WI_out + SI_out, start_dim=1)  # (N, 49 * hidden_size)
        out = self.dropout1(out)  # (N, hidden_size)
        out = F.normalize(input=out, p=2, dim=-1)  # (N, hidden_size)
        out = torch.relu(input=out)  # (N, hidden_size)
        out = self.fc2(out)  # (N, hidden_size)
        out = self.dropout2(out)  # (N, hidden_size)
        out = F.normalize(input=out, p=2, dim=-1)  # (N, hidden_size)
        out = torch.relu(input=out)  # (N, hidden_size)
        out = self.fc3(out)  # (N, num_classes)

        return out  # (N, num_classes)
