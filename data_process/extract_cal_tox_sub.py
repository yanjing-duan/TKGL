import sys
sys.path.append("../")

from smash import CircularLearner, PathLearner, FunctionGroupLearner
from rdkit import Chem
import pandas as pd


def safe_fit(learner, mols, labels, learner_name):
    try:
        _, matrix = learner.fit(
            mols, labels, aimLabel=1, 
            minNum=5, pCutoff=0.05, 
            accCutoff=0.7, Bonferroni=True
        )
        return matrix, learner
    except AttributeError as e:
        if "no attribute 'SMARTS'" in str(e):
            print(f"Warning: {learner_name} returned empty results, creating an empty placeholder matrix.")
            return pd.DataFrame(index=range(len(mols))), None
        else:
            raise

def extract_sig_sub(df = None):
    mols = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
    labels = df['Label'].values.astype('int')

    cirLearner = CircularLearner(minRadius=1, maxRadius=4)
    sigCirMatrix, cirLearner = safe_fit(cirLearner, mols, labels, "CircularLearner")

    pathLearner = PathLearner(minPath=1, maxPath=7)
    sigPathMatrix, pathLearner = safe_fit(pathLearner, mols, labels, "PathLearner")
    
    fgLearner = FunctionGroupLearner()
    sigFgMatrix, fgLearner = safe_fit(fgLearner, mols, labels, "FunctionGroupLearner")

    print(sigCirMatrix.shape, sigPathMatrix.shape, sigFgMatrix.shape)

    merge_fp_Matrix = pd.concat([df.loc[:,['smiles','Label']], sigCirMatrix, sigPathMatrix, sigFgMatrix], axis=1)
    
    return cirLearner, pathLearner, fgLearner, merge_fp_Matrix


if __name__ == "__main__":
    dataset_list = ['Carcinogenicity', 'H-HT']
    
    for endpoint in dataset_list:
        print("Processing endpoint:", endpoint)
        for seed in [7,17,27,37,47,57,67,77,87,97]:
            train_data = pd.read_csv("../split_data/{}/{}/train.csv".format(endpoint, seed))
            cirModel, pathModel, fgModel, calSA_train = extract_sig_sub(df = train_data)
            calSA_train.to_csv("../split_data/{}/{}/train_calSA.csv".format(endpoint, seed), index=False)

            if cirModel is not None:
                cirModel.saveModel('./models-for-calSA/cirLearner_{}_{}split.pkl'.format(endpoint, seed))
            if pathModel is not None:
                pathModel.saveModel('./models-for-calSA/pathLearner_{}_{}split.pkl'.format(endpoint, seed))
            if fgModel is not None:
                fgModel.saveModel('./models-for-calSA/fgLearner_{}_{}split.pkl'.format(endpoint, seed))
