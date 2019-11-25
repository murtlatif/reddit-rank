import torch
import torch.optim as optim
import torch.nn as nn
import torchtext
from torchtext import data
import spacy
import time
import argparse
import math
from sklearn.metrics import confusion_matrix

from models import *
from utils import *

class ModelTrainer:
    def __init__(self, args):
        torch.manual_seed(0)

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

        # Getting device to run on
        self.device = self.__get_default_device()

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

        # Move model to device selected
        self.model.to(self.device, non_blocking=True)

        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fnc = nn.CrossEntropyLoss()

        # Training tracking variables, reset on each call to train_validation_loop
        self.train_accuracies = []
        self.train_losses = []
        self.valid_accuracies = []
        self.valid_losses = []
        self.total_time = 0
        self.max_train_accuracy = (0, 0) # (value, epoch number)
        self.max_valid_accuracy = (0, 0)
        self.test_accuracies = 0

        # Overfit tracking variables, reset on each call to overfit_loop
        self.overfit_accuracies = []
        self.overfit_losses = []
        self.overfit_total_time = 0

        # Calculate number of batches
        self.num_batches = math.ceil(self.n_train / self.batch_size)
        self.num_batches_valid = math.ceil(self.n_valid / self.batch_size)
        self.num_batches_test = math.ceil(self.n_valid / self.batch_size)

    # - Public Methods -------------------------------------------------------------------------------------------------

    def train_validation_loop(self):
        self.__zero_tracking_vars()

        # Timing
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            tr_loss, tr_accuracy = self.__train_on_batches()
            v_loss, v_accuracy = self.__eval_validation()

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
        batch_index = 1
        num_correct = 0

        self.model.eval()
        with torch.no_grad():
            while(batch_index < self.num_batches_test):
                # Free GPU memory
                torch.cuda.empty_cache()

                batch = next(iter(self.test_iter))

                # Get batch data, send to GPU
                batch_titles = self.__to_GPU(batch.title[0])
                batch_title_lengths = self.__to_GPU(batch.title[1])
                batch_context = self.__get_context(batch)
                batch_scores = self.__to_GPU(batch.score)

                out = self.model(batch_titles,
                                 batch_context,
                                 batch_title_lengths).squeeze()

                num_correct += calc_num_correct(out, batch_scores)

                batch_index += 1

        self.test_accuracies = num_correct / self.n_test

    def overfit_loop(self):
        self.__zero_overfit_vars()

        # Timing
        start_time = time.time()

        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            overfit_loss_per_epoch = 0
            overfit_accuracy_per_epoch = 0

            batch = next(iter(self.overfit_iter))

            # Get batch data, send to GPU
            batch_titles = self.__to_GPU(batch.title[0])
            batch_title_lengths = self.__to_GPU(batch.title[1])
            batch_context = self.__get_context(batch)
            batch_scores = self.__to_GPU(batch.score)

            self.optimizer.zero_grad()

            out = self.model(batch_titles,
                             batch_context,
                             batch_title_lengths).squeeze()

            loss = self.loss_fnc(out, batch_scores)
            loss.backward()
            self.optimizer.step()

            # Record keeping
            overfit_loss_per_epoch += loss.cpu().detach().numpy()
            overfit_accuracy_per_epoch += calc_accuracy(out, batch_scores)

            # Record keeping
            self.overfit_losses.append(overfit_loss_per_epoch)
            self.overfit_accuracies.append(overfit_accuracy_per_epoch)

        # Timing
        self.overfit_total_time = time.time() - start_time

    def save_model(self, save_path):
        print("Model saved as:", save_path)
        torch.save(self.model, save_path)

    # - Private methods ------------------------------------------------------------------------------------------------

    def __eval_validation(self):
        batch_index = 1
        valid_loss_per_epoch = 0
        num_correct = 0

        self.model.eval()
        with torch.no_grad():
            while (batch_index <= self.num_batches_valid):
                # Free GPU memory for next batch
                torch.cuda.empty_cache()

                batch = next(iter(self.valid_iter))

                # Get batch data, send to GPU
                batch_titles = self.__to_GPU(batch.title[0])
                batch_title_lengths = self.__to_GPU(batch.title[1])
                batch_context = self.__get_context(batch)
                batch_scores = self.__to_GPU(batch.score)

                out = self.model(batch_titles,
                                 batch_context,
                                 batch_title_lengths).squeeze()

                loss = self.loss_fnc(out, batch_scores)

                # Record keeping
                valid_loss_per_epoch += loss.cpu().detach().numpy()
                num_correct += calc_num_correct(out, batch_scores)

                batch_index += 1
                del(loss) # Free up GPU space

        valid_accuracy_per_epoch = num_correct / self.n_valid

        return (valid_loss_per_epoch / self.num_batches_valid,
                valid_accuracy_per_epoch)

    def __train_on_batches(self):
        batch_index = 1
        train_loss_per_epoch = 0
        num_correct = 0

        self.model.train()
        while (batch_index <= self.num_batches):
            # Free GPU memory for next batch
            torch.cuda.empty_cache()

            batch = next(iter(self.train_iter))

            # Get batch data, send to GPU
            batch_titles = self.__to_GPU(batch.title[0])
            batch_title_lengths = self.__to_GPU(batch.title[1])
            batch_context = self.__get_context(batch)
            batch_scores = self.__to_GPU(batch.score)

            self.optimizer.zero_grad()

            out = self.model(batch_titles,
                             batch_context,
                             batch_title_lengths).squeeze()

            loss = self.loss_fnc(out, batch_scores)
            loss.backward()
            self.optimizer.step()

            # Record keeping
            train_loss_per_epoch += loss.cpu().detach().numpy()
            num_correct += calc_num_correct(out, batch_scores)

            batch_index += 1
            del(loss) # Free up GPU space

        train_accuracy_per_epoch = num_correct / self.n_train

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

    def __zero_overfit_vars(self):
        self.overfit_accuracies = []
        self.overfit_losses = []
        self.overfit_total_time = 0

    def __get_context(self, batch):
        batch_serious = batch.serious.unsqueeze(1)
        batch_nsfw = batch.nsfw.unsqueeze(1)
        batch_hours = torch.cat([
            batch.hour0.unsqueeze(1),
            batch.hour1.unsqueeze(1),
            batch.hour2.unsqueeze(1),
            batch.hour3.unsqueeze(1),
            batch.hour4.unsqueeze(1),
            batch.hour5.unsqueeze(1),
            batch.hour6.unsqueeze(1),
            batch.hour7.unsqueeze(1),
            batch.hour8.unsqueeze(1),
            batch.hour9.unsqueeze(1),
            batch.hour10.unsqueeze(1),
            batch.hour11.unsqueeze(1),
            batch.hour12.unsqueeze(1),
            batch.hour13.unsqueeze(1),
            batch.hour14.unsqueeze(1),
            batch.hour15.unsqueeze(1),
            batch.hour16.unsqueeze(1),
            batch.hour17.unsqueeze(1),
            batch.hour18.unsqueeze(1),
            batch.hour19.unsqueeze(1),
            batch.hour20.unsqueeze(1),
            batch.hour21.unsqueeze(1),
            batch.hour22.unsqueeze(1),
            batch.hour23.unsqueeze(1)], dim=1)
        batch_days = torch.cat([
            batch.mon.unsqueeze(1),
            batch.tue.unsqueeze(1),
            batch.wed.unsqueeze(1),
            batch.thu.unsqueeze(1),
            batch.fri.unsqueeze(1),
            batch.sat.unsqueeze(1),
            batch.sun.unsqueeze(1)], dim=1)
        return self.__to_GPU(torch.cat([batch_serious, batch_nsfw, batch_hours, batch_days], dim=1).float())

    def __load_title_data(self, path ='data/',
                          train_file = 'train.csv',
                          valid_file = 'valid.csv',
                          test_file = 'test.csv',
                          overfit_file = 'overfit.csv',
                          use_csv = True):

        # String
        TITLES = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)

        # Numerical (0 or 1)
        NSFW = data.Field(sequential=False, use_vocab=False)
        SERIOUS = data.Field(sequential=False, use_vocab=False)
        HOUR0 = data.Field(sequential=False, use_vocab=False)
        HOUR1 = data.Field(sequential=False, use_vocab=False)
        HOUR2 = data.Field(sequential=False, use_vocab=False)
        HOUR3 = data.Field(sequential=False, use_vocab=False)
        HOUR4 = data.Field(sequential=False, use_vocab=False)
        HOUR5 = data.Field(sequential=False, use_vocab=False)
        HOUR6 = data.Field(sequential=False, use_vocab=False)
        HOUR7 = data.Field(sequential=False, use_vocab=False)
        HOUR8 = data.Field(sequential=False, use_vocab=False)
        HOUR9 = data.Field(sequential=False, use_vocab=False)
        HOUR10 = data.Field(sequential=False, use_vocab=False)
        HOUR11 = data.Field(sequential=False, use_vocab=False)
        HOUR12 = data.Field(sequential=False, use_vocab=False)
        HOUR13 = data.Field(sequential=False, use_vocab=False)
        HOUR14 = data.Field(sequential=False, use_vocab=False)
        HOUR15 = data.Field(sequential=False, use_vocab=False)
        HOUR16 = data.Field(sequential=False, use_vocab=False)
        HOUR17 = data.Field(sequential=False, use_vocab=False)
        HOUR18 = data.Field(sequential=False, use_vocab=False)
        HOUR19 = data.Field(sequential=False, use_vocab=False)
        HOUR20 = data.Field(sequential=False, use_vocab=False)
        HOUR21 = data.Field(sequential=False, use_vocab=False)
        HOUR22 = data.Field(sequential=False, use_vocab=False)
        HOUR23 = data.Field(sequential=False, use_vocab=False)
        MON = data.Field(sequential=False, use_vocab=False)
        TUE = data.Field(sequential=False, use_vocab=False)
        WED = data.Field(sequential=False, use_vocab=False)
        THU = data.Field(sequential=False, use_vocab=False)
        FRI = data.Field(sequential=False, use_vocab=False)
        SAT = data.Field(sequential=False, use_vocab=False)
        SUN = data.Field(sequential=False, use_vocab=False)

        # Score
        LABELS = data.Field(sequential=False, use_vocab=False)

        self.train_data, self.valid_data, self.test_data = data.TabularDataset.splits(
            path=path,
            train=train_file, validation=valid_file, test=test_file,
            format='csv' if (use_csv) else 'tsv',
            skip_header=True, fields=[('title', TITLES),
                                      ('nsfw', NSFW),
                                      ('serious', SERIOUS),
                                      ('score', LABELS),
                                      ('hour0', HOUR0),
                                      ('hour1', HOUR1),
                                      ('hour2', HOUR2),
                                      ('hour3', HOUR3),
                                      ('hour4', HOUR4),
                                      ('hour5', HOUR5),
                                      ('hour6', HOUR6),
                                      ('hour7', HOUR7),
                                      ('hour8', HOUR8),
                                      ('hour9', HOUR9),
                                      ('hour10', HOUR10),
                                      ('hour11', HOUR11),
                                      ('hour12', HOUR12),
                                      ('hour13', HOUR13),
                                      ('hour14', HOUR14),
                                      ('hour15', HOUR15),
                                      ('hour16', HOUR16),
                                      ('hour17', HOUR17),
                                      ('hour18', HOUR18),
                                      ('hour19', HOUR19),
                                      ('hour20', HOUR20),
                                      ('hour21', HOUR21),
                                      ('hour22', HOUR22),
                                      ('hour23', HOUR23),
                                      ('mon', MON),
                                      ('tue', TUE),
                                      ('wed', WED),
                                      ('thu', THU),
                                      ('fri', FRI),
                                      ('sat', SAT),
                                      ('sun', SUN)])

        self.overfit_data = data.TabularDataset(
            path=path + overfit_file,
            format='csv' if (use_csv) else 'tsv',
            skip_header=True, fields=[('title', TITLES),
                                      ('nsfw', NSFW),
                                      ('serious', SERIOUS),
                                      ('score', LABELS),
                                      ('hour0', HOUR0),
                                      ('hour1', HOUR1),
                                      ('hour2', HOUR2),
                                      ('hour3', HOUR3),
                                      ('hour4', HOUR4),
                                      ('hour5', HOUR5),
                                      ('hour6', HOUR6),
                                      ('hour7', HOUR7),
                                      ('hour8', HOUR8),
                                      ('hour9', HOUR9),
                                      ('hour10', HOUR10),
                                      ('hour11', HOUR11),
                                      ('hour12', HOUR12),
                                      ('hour13', HOUR13),
                                      ('hour14', HOUR14),
                                      ('hour15', HOUR15),
                                      ('hour16', HOUR16),
                                      ('hour17', HOUR17),
                                      ('hour18', HOUR18),
                                      ('hour19', HOUR19),
                                      ('hour20', HOUR20),
                                      ('hour21', HOUR21),
                                      ('hour22', HOUR22),
                                      ('hour23', HOUR23),
                                      ('mon', MON),
                                      ('tue', TUE),
                                      ('wed', WED),
                                      ('thu', THU),
                                      ('fri', FRI),
                                      ('sat', SAT),
                                      ('sun', SUN)])

        self.TITLES = TITLES # Saved in order to build vocab later

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)
        self.n_test = len(self.test_data)
        self.n_overfit = len(self.overfit_data)

    def __create_data_iters(self):
        self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_sizes=(self.batch_size, self.batch_size, self.batch_size),
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
        self.TITLES.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
        self.vocab = self.TITLES.vocab

    def __get_default_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def __to_GPU(self, data):
        if (isinstance(data, (list,tuple))):
            return [self.__to_GPU(x, self.device) for x in data]
        return data.to(self.device, non_blocking=True)