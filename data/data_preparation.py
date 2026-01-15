''''df = pd.read_csv('df_with_sequence.csv')

protein = df["prot_base"].unique()


df_train = df[df["prot_base"].isin(train_p)]
df_val = df[df["prot_base"].isin(val_p)]
df_test = df[df["prot_base"].isin(test_p)]

train_ds = QmeanDataset(df_train)'''


q = ddf["Avg_Local_Score"].values.astype("float32")


qmin, qmax = q.min(), q.max()


def denorm(y_norm, qmin, qmax):
    return ((y_norm + 1) / 2) * (qmax - qmin) + qmin





