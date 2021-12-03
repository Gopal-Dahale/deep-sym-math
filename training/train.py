from argparse import ArgumentParser
from deep_sym_math.data.sym_data_module import SymDataModule
import warnings
from deep_sym_math.models.transformer import TransformerModel
from deep_sym_math.lit_models.base import BaseLitModel
import torch
import pytorch_lightning as pl
import wandb

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
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-elements", type=int, default=2)
    parser.add_argument("--fast-dev-run", type=bool, default=False)
    parser.add_argument('--n_enc_layers', type=int, default=6)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    num_epochs = args.num_epochs
    max_elements = args.max_elements
    fast_dev_run = args.fast_dev_run
    n_enc_layers = args.n_enc_layers
    n_dec_layers = args.n_dec_layers
    datasets = args.datasets

    data = SymDataModule(datasets, max_elements)
    model = TransformerModel(data.data_config(), {
        'n_enc_layers': n_enc_layers,
        'n_dec_layers': n_dec_layers
    })

    lit_model_class = BaseLitModel
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint,
                                                         model=model)
    else:
        lit_model = lit_model_class(model=model)

    gpus = None  # CPU
    if torch.cuda.is_available():
        gpus = -1  # all available GPUs

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss",
                                                         mode="min",
                                                         patience=10)
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
        monitor="val_loss",
        mode="min")

    # logger = pl.loggers.WandbLogger()
    # logger.watch(model)
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    trainer = pl.Trainer(
        gpus=gpus,
        fast_dev_run=fast_dev_run,
        max_epochs=num_epochs,
        callbacks=callbacks,
        #  logger=logger,
        weights_save_path='training/logs',
        weights_summary='full')

    trainer.tune(
        model=lit_model,
        datamodule=data,
    )

    trainer.fit(model=lit_model, datamodule=data)
    # trainer.test(model=lit_model, datamodule=data)


if __name__ == "__main__":
    main()