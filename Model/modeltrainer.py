import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
from torchtext import data
import spacy
import time
import argparse
import math

from models import *
from utils import *

class ModelTrainer:
    def __init__(self, args):
        # Load in hyperparameters
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.num_epochs = args.epochs
        self.model_type = args.model
        self.emb_dim = args.emb_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.num_kernels = args.num_kernels
        self.save_path = args.save_path
        self.data_path = args.data_path

        # Load data, create iterators and vocab object
        self.__load_title_data(path=self.data_path)
        self.__create_data_iters()
        self.__create_vocab_obj()

        # Create model
        self.model = Baseline(self.emb_dim, self.vocab)
        if self.model_type == 'baseline':
            print("Baseline")
        elif self.model_type == 'cnn':
            print("CNN")
            self.model = CNN(self.emb_dim, self.vocab, self.num_kernels)
        elif self.model_type == 'rnn':
            print("RNN")
            self.model = RNN(self.emb_dim, self.vocab, self.rnn_hidden_dim)

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fnc = nn.BCEWithLogitsLoss()

        # Training tracking variables, reset on each call to train_validation_loop
        self.train_accuracies = []
        self.train_losses = []
        self.valid_accuracies = []
        self.valid_losses = []
        self.total_time = 0
        self.max_train_accuracy = (0, 0) # (value, epoch number)
        self.max_valid_accuracy = (0, 0)
        self.test_accuracies = 0
        self.last_train_guesses = []
        self.last_train_answers = []
        self.last_valid_guesses = []
        self.last_valid_answers = []

        # Overfit tracking variables, reset on each call to overfit_loop
        self.overfit_accuracies = []
        self.overfit_losses = []
        self.overfit_total_time = 0

        # Calculate number of batches
        self.num_batches = math.ceil(self.n_train / self.batch_size)

    # - Public Methods -------------------------------------------------------------------------------------------------

    def train_validation_loop(self):
        self.__zero_tracking_vars()

        # Timing
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            is_last_batch = (epoch == self.num_epochs)
            tr_loss, tr_accuracy = self.__train_on_batches(is_last_batch)
            v_loss, v_accuracy = self.__eval_validation(is_last_batch)

            # Update min and maxes
            self.max_train_accuracy = (tr_accuracy, epoch) if (tr_accuracy > self.max_train_accuracy[0]) else self.max_train_accuracy
            self.max_valid_accuracy = (v_accuracy, epoch) if (v_accuracy > self.max_valid_accuracy[0]) else self.max_valid_accuracy

            # Record keeping
            self.train_losses.append(tr_loss)
            self.train_accuracies.append(tr_accuracy)
            self.valid_losses.append(v_loss)
            self.valid_accuracies.append(v_accuracy)

            print("Epoch " + str(epoch) + " Training accuracy: " + str(tr_accuracy) + " Validation accuracy: " + str(v_accuracy))

        # Timing
        self.total_time = time.time() - start_time

    def eval_test(self):
        batch = next(iter(self.test_iter))

        batch_titles, batch_title_lengths = batch.title
        batch_scores = batch.score
        batch_serious = batch.serious
        batch_spoiler = batch.spoiler
        batch_nsfw = batch.nsfw

        out = self.model(batch_titles,
                         batch_serious,
                         batch_spoiler,
                         batch_nsfw,
                         batch_title_lengths).squeeze()

        self.test_accuracies = calc_accuracy(out, batch_scores)

    def overfit_loop(self):
        self.__zero_overfit_vars()

        # Timing
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            overfit_loss_per_epoch = 0
            overfit_accuracy_per_epoch = 0

            batch = next(iter(self.overfit_iter))

            # Get batch data
            batch_titles, batch_title_lengths = batch.title
            batch_scores = batch.score
            batch_serious = batch.serious
            batch_spoiler = batch.spoiler
            batch_nsfw = batch.nsfw

            self.optimizer.zero_grad()
            out = self.model(batch_titles,
                             batch_serious,
                             batch_spoiler,
                             batch_nsfw,
                             batch_title_lengths).squeeze()
            loss = self.loss_fnc(out, batch_scores.float())
            loss.backward()
            self.optimizer.step()

            # Record keeping
            overfit_loss_per_epoch += loss.detach().numpy()
            overfit_accuracy_per_epoch += calc_accuracy(out, batch_scores)

            # Record keeping
            self.overfit_losses.append(overfit_loss_per_epoch)
            self.overfit_accuracies.append(overfit_accuracy_per_epoch)

            print("Epoch " + str(epoch) + " Training accuracy: " + str(overfit_accuracy_per_epoch))

        # Timing
        self.overfit_total_time = time.time() - start_time

    def save_model(self, save_path):
        print("Model saved as:", save_path)
        torch.save(self.model, save_path)

    # - Private methods ------------------------------------------------------------------------------------------------

    def __eval_validation(self, is_last_epoch = False):
        valid_loss_per_epoch = 0
        valid_accuracy_per_epoch = 0

        with torch.no_grad():
            batch = next(iter(self.valid_iter))

            # Get batch data
            batch_titles, batch_title_lengths = batch.title
            batch_scores = batch.score
            batch_serious = batch.serious
            batch_spoiler = batch.spoiler
            batch_nsfw = batch.nsfw

            out = self.model(batch_titles,
                             batch_serious,
                             batch_spoiler,
                             batch_nsfw,
                             batch_title_lengths).squeeze()
            loss = self.loss_fnc(out.squeeze(), batch_scores.float())

            # Record keeping
            valid_loss_per_epoch += loss.detach().numpy()
            valid_accuracy_per_epoch += calc_accuracy(out, batch_scores)

        if is_last_epoch:
            self.last_valid_answers = batch_scores.detach().numpy()
            self.last_valid_guesses = out.detach().numpy()

        return (valid_loss_per_epoch,
                valid_accuracy_per_epoch)

    def __train_on_batches(self, is_last_epoch = False):
        batch_index = 1
        train_loss_per_epoch = 0
        epoch_guesses = torch.tensor([], dtype=torch.float)
        epoch_answers = torch.tensor([], dtype=torch.float)
        while (batch_index <= self.num_batches):
            batch = next(iter(self.train_iter))

            # Get batch data
            batch_titles, batch_title_lengths = batch.title
            batch_scores = batch.score
            batch_serious = batch.serious
            batch_spoiler = batch.spoiler
            batch_nsfw = batch.nsfw

            self.optimizer.zero_grad()

            out = self.model(batch_titles,
                             batch_serious,
                             batch_spoiler,
                             batch_nsfw,
                             batch_title_lengths).squeeze()

            loss = self.loss_fnc(out.squeeze(), batch_scores.float())
            loss.backward()
            self.optimizer.step()

            # Record keeping
            train_loss_per_epoch += loss.detach().numpy()
            epoch_guesses = torch.cat([epoch_guesses, out.float()])
            epoch_answers = torch.cat([epoch_answers, batch_scores.float()])

            batch_index += 1

        train_accuracy_per_epoch = calc_accuracy(epoch_guesses, epoch_answers)

        if is_last_epoch:
            self.last_train_guesses = epoch_guesses.detach().numpy()
            self.last_train_answers = epoch_answers.detach().numpy()

        return (train_loss_per_epoch/self.num_batches,
                train_accuracy_per_epoch)

    def __zero_tracking_vars(self):
        self.train_accuracies = []
        self.train_losses = []
        self.valid_accuracies = []
        self.valid_losses = []
        self.total_time = 0
        self.max_train_accuracy = (0, 0)  # (value, epoch number)
        self.max_valid_accuracy = (0, 0)
        self.test_accuracies = 0
        self.last_train_guesses = []
        self.last_train_answers = []
        self.last_valid_guesses = []
        self.last_valid_answers = []

    def __zero_overfit_vars(self):
        self.overfit_accuracies = []
        self.overfit_losses = []
        self.overfit_total_time = 0

    def __load_title_data(self, path ='data/',
                          train_file = 'train.csv',
                          valid_file = 'valid.csv',
                          test_file = 'test.csv',
                          overfit_file = 'overfit.csv',
                          use_csv = True):

        # Strings and bools
        TITLES = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
        SERIOUS = data.Field(sequential=False, lower=True, tokenize='spacy')
        SPOILER = data.Field(sequential=False, lower=True, tokenize='spacy')
        NSFW = data.Field(sequential=False, lower=True, tokenize='spacy')

        # Numerical
        NUMCOMMENTS = data.Field(sequential=False, use_vocab=False)
        LABELS = data.Field(sequential=False, use_vocab=False) # Score of a post

        self.train_data, self.valid_data, self.test_data = data.TabularDataset.splits(
            path=path,
            train=train_file, validation=valid_file, test=test_file,
            format='csv' if (use_csv) else 'tsv',
            skip_header=True, fields=[('title', TITLES),
                                      ('serious', SERIOUS),
                                      ('num_comments', None),
                                      ('spoiler', SPOILER),
                                      ('nsfw', NSFW),
                                      ('UTC', None),
                                      ('score', LABELS)])

        self.overfit_data = data.TabularDataset(
            path=path + overfit_file,
            format='csv' if (use_csv) else 'tsv',
            skip_header=True, fields=[('title', TITLES),
                                      ('serious', SERIOUS),
                                      ('num_comments', None),
                                      ('spoiler', SPOILER),
                                      ('nsfw', NSFW),
                                      ('UTC', None),
                                      ('score', LABELS)])

        # In order to parse bool fields
        self.bool_data = data.TabularDataset(path="bool.csv", format='csv',
                                             skip_header=True, fields=[('title', TITLES)])

        self.TITLES = TITLES # Saved in order to build vocab later
        self.SERIOUS = SERIOUS
        self.NSFW = NSFW
        self.SPOILER = SPOILER

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)
        self.n_overfit = len(self.overfit_data)

    def __create_data_iters(self):
        self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_sizes=(self.batch_size, self.n_valid, self.n_test),
            sort_key=lambda x: len(x.title),
            device=None,
            sort_within_batch=True,
            repeat=False,
            shuffle=True)

        self.overfit_iter = data.BucketIterator(
            self.overfit_data,
            batch_size=self.n_overfit,
            sort_key=lambda x: len(x.title),
            device=None,
            sort_within_batch=True,
            repeat=False,
            shuffle=True)

    def __create_vocab_obj(self):
        self.TITLES.build_vocab(self.train_data, self.valid_data, self.overfit_data, self.test_data)
        self.SERIOUS.build_vocab(self.bool_data)
        self.NSFW.build_vocab(self.bool_data)
        self.SPOILER.build_vocab(self.bool_data)

        self.TITLES.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))

        self.vocab = self.TITLES.vocab