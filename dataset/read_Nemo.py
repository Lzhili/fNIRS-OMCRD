from collections import defaultdict
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

def get_sub_dict(dateset="empe"=):  # empe or afim
    csv_data_path = "./NEMO_data/" + dateset + "/"
    # Set this to the folder where you have the csv files
    epochs_df = pd.read_csv(csv_data_path + "epochs.csv", sep=";")
    epochs_metadata = pd.read_csv(csv_data_path + "epochs_metadata.csv", sep=";")

    X, y = defaultdict(list), defaultdict(list)
    chs = [c for c in epochs_df.columns if " hbo" in c or " hbr" in c]

    for epoch in epochs_df["epoch"].unique():  # 1203 or 720
        epoch_df = epochs_df[epochs_df["epoch"] == epoch]
        subject = epoch_df["subject"].iloc[0]

        X[subject].append(epoch_df[chs])
        if dateset == "empe":  # LANV:0, HANV:1, LAPV:2, HAPV:3
            y[subject].append(epochs_metadata.loc[epoch, "value"])
        else:  # afim_LANV:5, afim_HANV:6, afim_LAPV:7, afim_HAPV:8
            y[subject].append(epochs_metadata.loc[epoch, "value"] - 5)

    u, c = np.unique(np.concatenate([*y.values()]), return_counts=True)

    print("Created X and y for {} subjects."
          "\nX is a dictionary, key is subject id, value is list, and list elements are DataFrame."
          "\ny is a dictionary, key is subject id, value is list, and list elements are int(numpy)."
          "\nX size: {}"
          "\nclass counts: {}"
          .format(len(X), np.concatenate([*X.values()]).shape, dict(zip(u, c))))

    return X, y


class NemoDataset(Dataset):
    def __init__(self, X_dict, y_dict, sub_id, split_Hb=False, transform=True):
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.sub_id = sub_id
        self.split_Hb = split_Hb
        self.transform = transform
        feature = []
        label = []

        for sub in self.sub_id:
            feature.append(self.X_dict[sub])
            label.append(self.y_dict[sub])

        self.feature = torch.tensor(np.concatenate(feature).transpose((0, 2, 1)), dtype=torch.float)
        if self.split_Hb:
            HbO_feature = self.feature[:, 0::2, :]  # extract HbO channels (b, 24 ,600)
            HbR_feature = self.feature[:, 1::2, :]  # extract HbR channels
            HbO_feature = np.expand_dims(HbO_feature, axis=1)  # (b, 1, 24 ,600)
            HbR_feature = np.expand_dims(HbR_feature, axis=1)
            self.feature = np.concatenate((HbO_feature, HbR_feature), axis=1)  # (b, 2, 24 ,600)
            self.feature = torch.tensor(self.feature, dtype=torch.float)

        self.label = torch.tensor(np.concatenate(label), dtype=torch.long)
        print(self.feature.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        # z-score normalization
        if self.transform:
            mean, std = self.feature[item].mean(), self.feature[item].std()
            self.feature[item] = (self.feature[item] - mean) / std

        return self.feature[item], self.label[item]
