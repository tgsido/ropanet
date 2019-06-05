### args.py ###
import argparse

def get_main_args():
    """Get arguments needed in main.py."""
    parser = argparse.ArgumentParser('Entry point for ModaNetX')

    parser.add_argument("--do_train", action='store_true',help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--load_frozen_model", action='store_true', help="Whether to load frozen model")
    parser.add_argument("--frozen_model_dir", default=None, type=str, required=False,
                        help="The directory with frozen model and config")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")


    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=4, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    return args
