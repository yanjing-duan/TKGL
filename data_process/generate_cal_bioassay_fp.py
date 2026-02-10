import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl

from chemprop import data, featurizers, models


checkpoint_path = "./models-for-calBioassay/model_for_calBioassay.ckpt"
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)


dataset_list = ['Carcinogenicity', 'H-HT']


smiles_column = 'smiles'


features = pd.read_csv("./bioassay_features.csv").columns
bioassay_selected = pd.read_csv("./bioassay_features_selected.csv").columns.to_list()

for endpoint in dataset_list:
    for seed in [7,17,27,37,47,57,67,77,87,97]:
        df_train = pd.read_csv("../split_data/{}/{}/train.csv".format(endpoint, seed))
        df_val = pd.read_csv("../split_data/{}/{}/val.csv".format(endpoint, seed))
        df_test = pd.read_csv("../split_data/{}/{}/test.csv".format(endpoint, seed))

        train_data = [data.MoleculeDatapoint.from_smi(smi) for smi in df_train[smiles_column]]
        val_data = [data.MoleculeDatapoint.from_smi(smi) for smi in df_val[smiles_column]]
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in df_test[smiles_column]]

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        train_dset = data.MoleculeDataset(train_data, featurizer=featurizer)
        train_loader = data.build_dataloader(train_dset, shuffle=False)
        val_dset = data.MoleculeDataset(val_data, featurizer=featurizer)
        val_loader = data.build_dataloader(val_dset, shuffle=False)
        test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False)

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator="gpu",
                devices=1
            )
            train_preds = trainer.predict(mpnn, train_loader)
            val_preds = trainer.predict(mpnn, val_loader)
            test_preds = trainer.predict(mpnn, test_loader)

        train_preds = np.concatenate(train_preds, axis=0)
        train_preds = train_preds > 0.5
        train_preds = train_preds.astype(int)
        val_preds = np.concatenate(val_preds, axis=0)
        val_preds = val_preds > 0.5
        val_preds = val_preds.astype(int)
        test_preds = np.concatenate(test_preds, axis=0)
        test_preds = test_preds > 0.5
        test_preds = test_preds.astype(int)

        df_train = pd.concat([df_train, pd.DataFrame(train_preds, columns=features)], axis=1)
        df_val = pd.concat([df_val, pd.DataFrame(val_preds, columns=features)], axis=1)
        df_test = pd.concat([df_test, pd.DataFrame(test_preds, columns=features)], axis=1)
        df_train = df_train[['smiles', 'Label'] + bioassay_selected]
        df_val = df_val[['smiles', 'Label'] + bioassay_selected]
        df_test = df_test[['smiles', 'Label'] + bioassay_selected]

        df_train.to_csv("../split_data/{}/{}/train_calbioassay.csv".format(endpoint, seed), index=False)
        df_val.to_csv("../split_data/{}/{}/val_calbioassay.csv".format(endpoint, seed), index=False)
        df_test.to_csv("../split_data/{}/{}/test_calbioassay.csv".format(endpoint, seed), index=False)