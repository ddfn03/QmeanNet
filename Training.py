import torch

from DataPreparation import denorm, qmin, qmax, df_train, df_val, df_test
from ProtBerQmean import ProtBerQmean
from QmeanDataModule import QmeanDataModule
import pytorch_lightning as pl



dm = QmeanDataModule(df_train, df_val, df_test, batch_size=8, max_len=240)
dm.setup()

model = ProtBerQmean(lr=3e-4)

checkpoint_cb = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1
)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=5,
    enable_progress_bar=True,
    callbacks=[checkpoint_cb]
)

trainer.fit(model, dm)
trainer.save_checkpoint("protbert_qmean_final.ckpt")