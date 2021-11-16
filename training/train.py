from argparse import ArgumentParser
from re import A
from deep_sym_math.data.sym_data_module import SymDataModule
import warnings

warnings.filterwarnings('ignore')


def _setup_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Datasets to use',
        default=['prim_fwd'],
        choices=['prim_fwd', 'prim_bwd', 'prim_ibp', 'ode1', 'ode2'])
    parser.add_argument("--load_checkpoint", type=str, default=None)
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    datasets = args.datasets

    data = SymDataModule(datasets)
    data.prepare_data()
    data.setup()

    # it = data.train_dataloader()
    # for (x1, len1), (x2, len2), nb_ops in it:
    #     print(x1, x2, len1, len2, nb_ops)

    # (x1, len1), (x2, len2), nb_ops = next(iter))
    # print(x1)
    # print(x2)


if __name__ == "__main__":
    main()