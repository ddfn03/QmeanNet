import torch

from DataPreparation import denorm, qmin, qmax, df_train, df_val, df_test
from ProtBerQmean import ProtBerQmean
from QmeanDataModule import QmeanDataModule

dm = QmeanDataModule(df_train , df_val , df_test , batch_size=8 , max_len=240)
dm.setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_loader = dm.val_dataloader()

model = ProtBerQmean.load_from_checkpoint("protbert_qmean_final.ckpt")
model.eval()
model.to(device)

batch = next(iter(val_loader))
input_ids, attn_mask, y = batch
input_ids = input_ids.to(device)
attn_mask = attn_mask.to(device)
y = y.to(device)
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        input_ids, attn_mask, y = batch
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)

        y_pred = model(input_ids, attn_mask)

        print(f"Batch {i}")
        print("y:", y[:8])
        print("y_pred:", y_pred[:8])

        if i == 2:  # fermati dopo 3 batch, per non esplodere di output
            break

y_true = y
y_pred_denorm = y_pred

for t, p in list(zip(y_true.tolist(), y_pred_denorm.tolist()))[:10]:
    print(f"true={t:.3f}, pred={p:.3f}")