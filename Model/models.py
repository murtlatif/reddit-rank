import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):

    def __init__(self, emb_dim, vocab, classes):
        super(Baseline, self).__init__()
        num_context_vars = 2 + 24 + 7 # serious - nsfw - hour0 - ... - hour23 - mon - ... - sun

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Linear(emb_dim + num_context_vars, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, title, context, lengths=None):
        embedded = self.embedding(title)

        average = embedded.mean(0)

        withcontext = torch.cat([average, context], 1)

        output = torch.sigmoid(self.fc1(withcontext).squeeze(1))
        output = self.fc2(output)

        return output

class CNN(nn.Module):
    def __init__(self, emb_dim, vocab, n_filters, classes):
        super(CNN, self).__init__()
        num_context_vars = 2 + 24 + 7 # serious - nsfw - hour0 - ... - hour23 - mon - ... - sun

        self.n_filters = n_filters

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(emb_dim, 1))
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(emb_dim, 2))
        self.conv3 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(emb_dim, 3))
        self.conv4 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(emb_dim, 4))
        self.bn_conv = nn.BatchNorm2d(n_filters * 4)
        self.dr_conv = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear(n_filters * 4  + num_context_vars, 300)
        self.bn_fc1 = nn.BatchNorm1d(300)
        self.dr_fc1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(300, 100)
        self.bn_fc2 = nn.BatchNorm1d(100)
        self.dr_fc2 = nn.Dropout2d(0.5)
        self.fc3 = nn.Linear(100, classes)
        self.LeakyRelu = nn.LeakyReLU()

    def forward(self, title, context, lengths):

        batch_size = title.shape[1]
        max_length = torch.max(lengths)

        pool1 = nn.MaxPool2d((1, max_length-0), 1)  # k = 1
        pool2 = nn.MaxPool2d((1, max_length-1), 1)  # k = 2
        pool3 = nn.MaxPool2d((1, max_length-2), 1)  # k = 3
        pool4 = nn.MaxPool2d((1, max_length-3), 1)  # k = 4

        embedded = self.embedding(title)

        # Transform embedded tensor to match input to conv layers
        embedded = torch.unsqueeze(embedded, 1)
        embedded = torch.transpose(embedded, 0, 2)
        embedded = torch.transpose(embedded, 2, 3)

        # Ordering is CONV -> POOL -> RELU -> CONCATENATE -> BATCHNORM -> DROPOUT
        conv1out = self.LeakyRelu(pool1(self.conv1(embedded)))
        conv2out = self.LeakyRelu(pool2(self.conv2(embedded)))
        conv3out = self.LeakyRelu(pool3(self.conv3(embedded)))
        conv4out = self.LeakyRelu(pool4(self.conv4(embedded)))
        concatenated = torch.cat([conv1out, conv2out, conv3out, conv4out], dim=1)
        concatenated = self.dr_conv(self.bn_conv(concatenated))

        # Reshape output
        concatenated = torch.reshape(concatenated, (batch_size, 4 * self.n_filters))

        # Add context flags (booleans)
        withcontext = torch.cat([concatenated, context], 1)

        # MLP
        out = self.dr_fc1(self.LeakyRelu(self.bn_fc1(self.fc1(withcontext))))
        out = self.dr_fc2(self.LeakyRelu(self.bn_fc2(self.fc2(out))))
        out = self.fc3(out)

        return out

class RNN(nn.Module):
    def __init__(self, emb_dim, vocab, hidden_dim, classes):
        super(RNN, self).__init__()
        num_context_vars = 2 + 24 + 7  # serious - nsfw - hour0 - ... - hour23 - mon - ... - sun

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(emb_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + num_context_vars, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, title, context, lengths):
        embedded = self.embedding(title)
        gru_out, h = self.GRU(embedded)
        h = h.squeeze()

        withcontext = torch.cat([h, context], 1)

        output = torch.sigmoid(self.fc1(withcontext).squeeze(1))
        output = self.fc2(output)

        return out