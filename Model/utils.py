import torch
import numpy

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# guesses and answers are numpy arrays
def calc_squared_error(guesses, answers):
    errors = answers - guesses
    sqr_errors = errors * errors

    return sum(sqr_errors)

# guesses and answers are pytorch tensors
def calc_RSQR(guesses, answers):
    guesses = guesses.detach().numpy()
    answers = answers.detach().numpy()

    n = len(guesses)
    SE = calc_squared_error(guesses, answers)
    MSE = SE / n
    rMSE = MSE/numpy.var(answers)

    return 1 - rMSE

# guesses and answers are pytorch tensors
def calc_RMSE(guesses, answers):
    guesses = guesses.detach().numpy()
    answers = answers.detach().numpy()

    n = len(guesses)
    SE = calc_squared_error(guesses, answers)

    return SE ** 0.5

def plot_overfit_results(num_epochs, overfit_RSQRs, overfit_RMSEs, overfit_losses):

    fig, (loss_plot, RSQR_plot, RMSE_plot) = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.5)

    # Loss plot
    loss_plot.plot(range(1, num_epochs + 1), overfit_losses)
    loss_plot.axis([1, num_epochs, 0, max(overfit_losses) + min(overfit_losses)])
    loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_plot.set_ylabel("Loss")
    loss_plot.set_xlabel("Number of epochs")
    loss_plot.set_title("Loss")

    # RSQR plot
    RSQR_plot.plot(range(1, num_epochs + 1), overfit_RSQRs)
    RSQR_plot.axis([1, num_epochs, 0, 1 + 0.05])
    RSQR_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    RSQR_plot.set_ylabel("R Squared")
    RSQR_plot.set_xlabel("Number of epochs")
    RSQR_plot.set_title("R^2")

    # RMSE plot
    RMSE_plot.plot(range(1, num_epochs + 1), overfit_RMSEs)
    RMSE_plot.axis([1, num_epochs, 0, max(overfit_RMSEs) + min(overfit_RMSEs)])
    RMSE_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    RMSE_plot.set_ylabel("Root Mean Squared Error")
    RMSE_plot.set_xlabel("Number of epochs")
    RMSE_plot.set_title("RMSE")

    # Show plots
    plt.show()

# plots RSQRs, RMSEs and losses
def plot_results(num_epochs,
                 train_RSQRs, train_RMSEs, train_losses,
                 valid_RSQRs, valid_RMSEs, valid_losses):

    fig, (loss_plot, RSQR_plot, RMSE_plot) = plt.subplots(1, 3)
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

    # RSQR plot
    RSQR_plot.plot(range(1, num_epochs + 1), train_RSQRs)
    RSQR_plot.plot(range(1, num_epochs + 1), valid_RSQRs)
    RSQR_plot.axis([1, num_epochs, 0, 1 + 0.05])
    RSQR_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    RSQR_plot.set_ylabel("R Squared")
    RSQR_plot.set_xlabel("Number of epochs")
    RSQR_plot.set_title("R^2")
    RSQR_plot.legend(["Training", "Validation"])

    # RMSE plot
    RMSE_plot.plot(range(1, num_epochs + 1), train_RMSEs)
    RMSE_plot.plot(range(1, num_epochs + 1), valid_RMSEs)
    RMSE_plot.axis([1, num_epochs, 0, max(max(valid_RMSEs), max(train_RMSEs)) + min(train_RMSEs)])
    RMSE_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    RMSE_plot.set_ylabel("Root Mean Squared Error")
    RMSE_plot.set_xlabel("Number of epochs")
    RMSE_plot.set_title("RMSE")
    RMSE_plot.legend(["Training", "Validation"])

    # Show plots
    plt.show()

