import pandas as pd
import os
import re
import  torch
import torchvision
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transformers.utils.logging import enable_progress_bar

from ProtBerQmean import ProtBerQmean
from QmeanDataModule import QmeanDataModule
from QmeanDataset import QmeanDataset

df = pd.read_csv('df_with_sequence.csv')

protein = df["prot_base"].unique()
train_p , test_p = train_test_split(protein , test_size=0.2 , random_state=42)
train_p , val_p = train_test_split(train_p , test_size=0.2 , random_state=42)

df_train = df[df["prot_base"].isin(train_p)]
df_val = df[df["prot_base"].isin(val_p)]
df_test = df[df["prot_base"].isin(test_p)]

train_ds = QmeanDataset(df_train)


q = df["Avg_Local_Score"].values.astype("float32")
qmin, qmax = q.min(), q.max()


def denorm(y_norm, qmin, qmax):
    return ((y_norm + 1) / 2) * (qmax - qmin) + qmin





