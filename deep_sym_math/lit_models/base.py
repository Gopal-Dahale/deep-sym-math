import pytorch_lightning as pl

OPTIMIZER = "Adam"
LR = 1e-4
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100

class BaseLitModel(pl.LightningModule):
    