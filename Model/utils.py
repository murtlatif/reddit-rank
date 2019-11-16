import torch
import numpy
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def calc_accuracy(guesses, answers):
    guesses = guesses.detach()
    answers = answers.detach().numpy()
    n = len(answers)

    # Apply sigmoid to output
    sigmoid = nn.Sigmoid()
    guesses = sigmoid(guesses)
    guesses = guesses.numpy()
    guesses = numpy.round(guesses, 0)

    # Count number of equivalent elements
    res = numpy.sum(guesses == answers)

    return res / n

def plot_results(num_epochs, train_losses, train_accuracies, valid_losses, valid_accuracies):
    fig, (loss_plot, accuracy_plot) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.5)

    # Loss plot
    loss_plot.plot(range(1, num_epochs + 1), train_losses)
    loss_plot.plot(range(1, num_epochs + 1), valid_losses)
    loss_plot.axis([1, num_epochs, 0, max(max(valid_losses), max(train_losses)) + min(train_losses)])
    loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_plot.set_ylabel("Loss")
    loss_plot.set_xlabel("Number of epochs")
    loss_plot.set_title("Loss")
    loss_plot.legend(["Training", "Validation"])

    # Accuracy plot
    accuracy_plot.plot(range(1, num_epochs + 1), train_accuracies)
    accuracy_plot.plot(range(1, num_epochs + 1), valid_accuracies)
    accuracy_plot.axis([1, num_epochs, 0, 1 + 0.05])
    accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    accuracy_plot.set_ylabel("Accuracy")
    accuracy_plot.set_xlabel("Number of epochs")
    accuracy_plot.set_title("Accuracy")
    accuracy_plot.legend(["Training", "Validation"])

    # Show plots
    plt.show()

def plot_overfit_results(num_epochs, train_losses, train_accuracies):
    fig, (loss_plot, accuracy_plot) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.5)

    # Loss plot
    loss_plot.plot(range(1, num_epochs + 1), train_losses)
    loss_plot.axis([1, num_epochs, 0, max(train_losses) + min(train_losses)])
    loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_plot.set_ylabel("Loss")
    loss_plot.set_xlabel("Number of epochs")
    loss_plot.set_title("Loss")
    loss_plot.legend(["Training"])

    # Accuracy plot
    accuracy_plot.plot(range(1, num_epochs + 1), train_accuracies)
    accuracy_plot.axis([1, num_epochs, 0, 1 + 0.05])
    accuracy_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    accuracy_plot.set_ylabel("Accuracy")
    accuracy_plot.set_xlabel("Number of epochs")
    accuracy_plot.set_title("Accuracy")
    accuracy_plot.legend(["Training"])

    # Show plots
    plt.show()