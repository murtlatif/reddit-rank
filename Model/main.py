import argparse

from modeltrainer import *
from utils import *

def main(args):
    # Load in options
    save_model = args.save_model
    save_path = args.save_path
    overfit = args.overfit

    # Create model trainer
    m = ModelTrainer(args)

    # Overfitting
    if overfit:
        # Overfit model
        m.overfit_loop()

        # Statistics
        print("Total time: {0:.2f}s".format(m.overfit_total_time))

        # Plot
        plot_overfit_results(m.num_epochs,
                             m.overfit_losses,
                             m.overfit_accuracies)
    else:
        # Train model
        m.train_validation_loop()

        # Evaluate test set with model
        m.eval_test()

        # Statistics
        print("Total time: {0:.2f}s".format(m.total_time))
        print("Maximum training accuracy: " + str(m.max_train_accuracy[0]) + " at epoch: " + str(m.max_train_accuracy[1]))
        print("Maximum validation accuracy: " + str(m.max_valid_accuracy[0]) + " at epoch: " + str(m.max_valid_accuracy[1]))
        print("Test accuracy:", m.test_accuracies)

        # Confusion Matrix
        m.print_confusion_matrix()

        # Plot
        plot_results(m.num_epochs,
                     m.train_losses, m.train_accuracies,
                     m.valid_losses, m.valid_accuracies)
        # plot_guess_answers(m.last_train_guesses, m.last_train_answers, m.last_valid_guesses, m.last_valid_answers)

    # Saving
    if (save_model):
        m.save_model(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-kernels', type=int, default=50)
    parser.add_argument('--overfit', type=int, default=0)
    parser.add_argument('--save-model', type=int, default=0)
    parser.add_argument('--save-path', type=str, default='model.pt')
    parser.add_argument('--data-path', type=str, default='data/')

    args = parser.parse_args()

    main(args)

# FULL TERMINAL COMMAND:
"""
python main.py --model "rnn" --batch-size 64 --lr 0.1 --epochs 100 --emb-dim 100 --rnn-hidden-dim 100 --num-kernels 50 --overfit 0 --save-model 0 --save-path "placeholder.pt"
"""

# NO SAVE NO OVERFIT
"""
python main.py --model "rnn" --batch-size 64 --lr 0.1 --epochs 100 --emb-dim 100 --rnn-hidden-dim 100 --num-kernels 50 --data-path "data/"
"""