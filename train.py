import torch

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from model.protbert_qmean import ProtBerQmean
from data.qmean_datamodule import QmeanDataModule

'''class MyCLI(LightningCLI):
    @rank_zero_only
    def after_instantiate_classes(self, **kwargs):
        model = self.model
        logger = self.trainer.logger

        if isinstance(logger, WandbLogger):
            logger.watch(
                model=model,
                log="all",  # gradients + parameter histograms + models topology
                log_freq=100,
                log_graph=False,  # in case models graph is large or breaks wandb
            )
            logger.experiment.config.update(dict(self.config))
            self.save_config_kwargs["config_filename"] = os.path.join(self.trainer.default_root_dir, logger.experiment.id,
                                                                  "strokeformer_config.yaml")
            os.makedirs(os.path.join(self.trainer.default_root_dir, logger.experiment.id), exist_ok=True)
            # reinstantiate trainer
            extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
            trainer_config = {**self._get(self.config_init, "trainer", default={}), **kwargs}
            self.trainer = self._instantiate_trainer(trainer_config, extra_callbacks) # noqa'''

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    LightningCLI(
        model_class=ProtBerQmean,
        datamodule_class=QmeanDataModule,
        save_config_kwargs={"overwrite": True},
    )
