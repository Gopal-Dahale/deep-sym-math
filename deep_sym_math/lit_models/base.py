import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn.functional as F

OPTIMIZER = "Adam"
LR = 1e-4
ONE_CYCLE_TOTAL_STEPS = 100


class Accuracy(torchmetrics.Metric):
    """Accuracy Metric with a hack."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("n_valid", default=torch.zeros(1000, dtype=torch.long))  # pylint: disable=not-callable
        self.add_state("n_total", default=torch.zeros(1000, dtype=torch.long))  # pylint: disable=not-callable

    def update(self, target, pred_mask, scores, len_target, nb_ops):
        # Update metric states
        # Correct outputs per sequence / valid top-1 predictions
        t = torch.zeros_like(pred_mask).type_as(pred_mask)
        t[pred_mask] += scores.max(1)[1] == target
        valid = (t.sum(0) == len_target - 1).long()

        # Stats
        self.n_valid.index_add_(-1, nb_ops, valid)
        self.n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

    def compute(self):
        # Compute final result
        _n_valid = self.n_valid.sum().item()
        _n_total = self.n_total.sum().item()
        return _n_valid / _n_total


class BaseLitModel(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.args = {}
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps",
                                                   ONE_CYCLE_TOTAL_STEPS)
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (x, len_x), (y, len_y), _ = batch

        # Target words to predict
        alen = torch.arange(len_y.max(), dtype=torch.long).type_as(len_y)

        # Do not predict anything given the last target word
        pred_mask = alen[:, None] < len_y[None] - 1

        y_masked = y[1:].masked_select(pred_mask[:-1])
        assert len(y_masked) == (len_y - 1).sum().item()

        # Forward / Loss
        logits = self.model(x, len_x, y, len_y)
        _, train_loss = self.loss_fn(logits, y_masked, pred_mask)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument

        (x, len_x), (y, len_y), nb_ops = batch

        # Target words to predict
        alen = torch.arange(len_y.max(), dtype=torch.long).type_as(len_y)

        # Do not predict anything given the last target word
        pred_mask = alen[:, None] < len_y[None] - 1
        y_masked = y[1:].masked_select(pred_mask[:-1])
        assert len(y_masked) == (len_y - 1).sum().item()

        # Forward / Loss
        logits = self.model(x, len_x, y, len_y)
        scores, val_loss = self.loss_fn(logits, y_masked, pred_mask)
        self.log("val_loss",
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        acc = self.val_acc(y_masked, pred_mask, scores, len_y, nb_ops)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (x, len_x), (y, len_y), nb_ops = batch

        # Target words to predict
        alen = torch.arange(len_y.max(), dtype=torch.long)

        # Do not predict anything given the last target word
        pred_mask = alen[:, None] < len_y[None] - 1

        y_masked = y[1:].masked_select(pred_mask[:-1])
        assert len(y_masked) == (len_y - 1).sum().item()

        # Forward / Loss
        logits = self.model(x, len_x, y, len_y)
        scores, test_loss = self.loss_fn(logits, y_masked, pred_mask)
        self.log("test_loss",
                 test_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        acc = self.test_acc(y_masked, pred_mask, scores, len_y, nb_ops)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def loss_fn(self, logits, y, pred_mask):
        x = logits[pred_mask.unsqueeze(-1).expand_as(logits)].view(
            -1, self.model.dim)
        assert (y == self.model.pad_index).sum().item() == 0
        scores = self.model.dec_proj(x).view(-1, self.model.n_words)
        loss = F.cross_entropy(scores, y, reduction='mean')
        return scores, loss