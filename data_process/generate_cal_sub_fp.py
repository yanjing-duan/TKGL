import os
import pandas as pd
from rdkit import Chem

import bz2
import pickle
import sys
sys.path.append("../")


def load_model(file):
    with bz2.BZ2File(file, 'r') as f:
        Learner_loaded = pickle.load(f)
    f.close()
    return Learner_loaded


if __name__ == "__main__":
    dataset_list = ['Carcinogenicity', 'H-HT']

    for endpoint in dataset_list:
        for seed in [7,17,27,37,47,57,67,77,87,97]:
            data_test = pd.read_csv("../split_data/{}/{}/test.csv".format(endpoint, seed))
            data_val = pd.read_csv("../split_data/{}/{}/val.csv".format(endpoint, seed))
            mols_test = [Chem.MolFromSmiles(smi) for smi in data_test['smiles']]
            mols_val = [Chem.MolFromSmiles(smi) for smi in data_val['smiles']]

            cirModel_path = './models-for-calSA/cirLearner_{}_{}split.pkl.pbz2'.format(endpoint, seed)
            if os.path.exists(cirModel_path):
                cirLearner_loaded = load_model(cirModel_path)
                cirPred_test, cirPredMatrix_test = cirLearner_loaded.predict(mols_test)
                cirPred_val, cirPredMatrix_val = cirLearner_loaded.predict(mols_val)
            else:
                print("Warning: cirLearner model not found for endpoint: {}, seed: {}".format(endpoint, seed))
                cirPredMatrix_test = pd.DataFrame(index=range(len(mols_test)))
                cirPredMatrix_val = pd.DataFrame(index=range(len(mols_val)))
            pathModel_path = './models-for-calSA/pathLearner_{}_{}split.pkl.pbz2'.format(endpoint, seed)
            if os.path.exists(pathModel_path):
                pathLearner_loaded = load_model(pathModel_path)
                pathPred_test, pathPredMatrix_test = pathLearner_loaded.predict(mols_test)
                pathPred_val, pathPredMatrix_val = pathLearner_loaded.predict(mols_val)
            else:
                print("Warning: pathLearner model not found for endpoint: {}, seed: {}".format(endpoint, seed))
                pathPredMatrix_test = pd.DataFrame(index=range(len(mols_test)))
                pathPredMatrix_val = pd.DataFrame(index=range(len(mols_val)))

            fgModel_path = './models-for-calSA/fgLearner_{}_{}split.pkl.pbz2'.format(endpoint, seed)
            if os.path.exists(fgModel_path):
                fgLearner_loaded = load_model(fgModel_path)
                fgPred_test, fgPredMatrix_test = fgLearner_loaded.predict(mols_test)
                fgPred_val, fgPredMatrix_val = fgLearner_loaded.predict(mols_val)
            else:
                print("Warning: fgLearner model not found for endpoint: {}, seed: {}".format(endpoint, seed))
                fgPredMatrix_test = pd.DataFrame(index=range(len(mols_test)))
                fgPredMatrix_val = pd.DataFrame(index=range(len(mols_val)))

            calSA_test = pd.concat([data_test.loc[:, ['smiles', 'Label']], cirPredMatrix_test, pathPredMatrix_test, fgPredMatrix_test], axis=1)
            calSA_val = pd.concat([data_val.loc[:, ['smiles', 'Label']], cirPredMatrix_val, pathPredMatrix_val, fgPredMatrix_val], axis=1)
            calSA_test.to_csv("../split_data/{}/{}/test_calSA.csv".format(endpoint, seed), index=False)
            calSA_val.to_csv("../split_data/{}/{}/val_calSA.csv".format(endpoint, seed), index=False)