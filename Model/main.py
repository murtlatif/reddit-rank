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

        # Plot
        plot_overfit_results(m.num_epochs,
                             m.overfit_RSQRs,
                             m.overfit_RMSEs,
                             m.overfit_losses)
    else:
        # Train model
        m.train_validation_loop()

        # Evaluate test set with model
        m.eval_test()

        # Plot
        plot_results(m.num_epochs,
                     m.train_RSQRs, m.train_RMSEs, m.train_losses,
                     m.valid_RSQRs, m.valid_RMSEs, m.valid_losses)

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