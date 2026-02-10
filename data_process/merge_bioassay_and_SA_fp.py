import pandas as pd


dataset_list = ['Carcinogenicity', 'H-HT']



for endpoint in dataset_list:
    for seed in [7,17,27,37,47,57,67,77,87,97]:
        SA_train = pd.read_csv("../split_data/{}/{}/train_calSA.csv".format(endpoint, seed))
        SA_val = pd.read_csv("../split_data/{}/{}/val_calSA.csv".format(endpoint, seed))
        SA_test = pd.read_csv("../split_data/{}/{}/test_calSA.csv".format(endpoint, seed))
        KE_train = pd.read_csv("../split_data/{}/{}/train_expcalbioassay.csv".format(endpoint, seed))
        KE_val = pd.read_csv("../split_data/{}/{}/val_calbioassay.csv".format(endpoint, seed))
        KE_test = pd.read_csv("../split_data/{}/{}/test_calbioassay.csv".format(endpoint, seed))

        SAKE_train = pd.merge(KE_train, SA_train, how='inner', on=['smiles', 'Label'])
        SAKE_val = pd.merge(KE_val, SA_val, how='inner', on=['smiles', 'Label'])
        SAKE_test = pd.merge(KE_test, SA_test, how='inner', on=['smiles', 'Label'])

        SAKE_train.to_csv("../split_data/{}/{}/train_BioassaySA.csv".format(endpoint, seed), index=False)
        SAKE_val.to_csv("../split_data/{}/{}/val_BioassaySA.csv".format(endpoint, seed), index=False)
        SAKE_test.to_csv("../split_data/{}/{}/test_BioassaySA.csv".format(endpoint, seed), index=False)
        
        print("Endpoint: {}, Seed: {}, SA_num: {}, KE_num: {}, SAKE_num: {}, train_size: {}, val_size: {}, test_size: {}".format(
            endpoint, seed, SA_train.shape[1] - 2, KE_train.shape[1] - 2, # -2 for 'smiles' and 'Label'
            SAKE_train.shape[1] - 2, SAKE_train.shape[0], SAKE_val.shape[0], SAKE_test.shape[0]))