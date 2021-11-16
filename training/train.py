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


if __name__ == "__main__":
    main()