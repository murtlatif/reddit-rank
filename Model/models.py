import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):

    def __init__(self, emb_dim, vocab):
        super(Baseline, self).__init__()
        num_context_vars = 3

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(emb_dim + num_context_vars, 1)

    # context = [serious, spoiler, nsfw, num_comments]
    def forward(self, title, serious, spoiler, nsfw, lengths=None):
        embedded = self.embedding(title)

        average = embedded.mean(0)
        serious = serious.unsqueeze(1).float()
        spoiler = spoiler.unsqueeze(1).float()
        nsfw = nsfw.unsqueeze(1).float()

        average_and_context = torch.cat([average, serious, spoiler, nsfw], 1)

        output = self.fc(average_and_context).squeeze(1)

        return output

class CNN(nn.Module):
    def __init__(self, emb_dim, vocab, n_filters, filter_sizes = (2, 4)):
        super(CNN, self).__init__()
        num_context_vars = 3

        filter_size1 = filter_sizes[0]
        filter_size2 = filter_sizes[1]

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(emb_dim, filter_size1))
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=n_filters,
                               kernel_size=(emb_dim, filter_size2))
        self.fc = nn.Linear(emb_dim + 3, 1)


    def forward(self, title, serious, spoiler, nsfw, lengths):

        batch_size = title.shape[1]
        max_length = max(lengths.detach().numpy())

        serious = serious.unsqueeze(1).float()
        spoiler = spoiler.unsqueeze(1).float()
        nsfw = nsfw.unsqueeze(1).float()

        pool1 = nn.MaxPool2d((1, max_length-1), 1)  # k = 2
        pool2 = nn.MaxPool2d((1, max_length-3), 1)  # k = 4

        embedded = self.embedding(title)

        # transform embedded tensor to match input to conv layers
        embedded = torch.unsqueeze(embedded, 1)
        embedded = torch.transpose(embedded, 0, 2)
        embedded = torch.transpose(embedded, 2, 3)

        conv1out = torch.relu(self.conv1(embedded))
        conv2out = torch.relu(self.conv2(embedded))

        conv1out_pooled = pool1(conv1out)
        conv2out_pooled = pool2(conv2out)

        concatenated = torch.cat([conv1out_pooled, conv2out_pooled], dim=1)
        concatenated = torch.reshape(concatenated, (batch_size, 100))

        withcontext = torch.cat([concatenated,
                                 serious,
                                 spoiler,
                                 nsfw], 1)

        out = self.fc(withcontext)

        return out

class RNN(nn.Module):
    def __init__(self, emb_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        num_context_vars = 3

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + num_context_vars, 1)

    def forward(self, title, serious, spoiler, nsfw, lengths):
        embedded = self.embedding(title)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        gru_out, h = self.GRU(embedded)
        h = h.squeeze()

        serious = serious.unsqueeze(1).float()
        spoiler = spoiler.unsqueeze(1).float()
        nsfw = nsfw.unsqueeze(1).float()

        withcontext = torch.cat([h, serious, spoiler, nsfw], 1)

        out = self.fc(withcontext)

        return out