import pandas as pd
import numpy as np
from utils import smiles2adjoin
import tensorflow as tf

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

"""     

{'O': 5000757, 'C': 34130255, 'N': 5244317, 'F': 641901, 'H': 37237224, 'S': 648962, 
'Cl': 373453, 'P': 26195, 'Br': 76939, 'B': 2895, 'I': 9203, 'Si': 1990, 'Se': 1860, 
'Te': 104, 'As': 202, 'Al': 21, 'Zn': 6, 'Ca': 1, 'Ag': 3}

H C N O F S  Cl P Br B I Si Se
"""

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
         'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}

num2str =  {i:j for j,i in str2num.items()}

def drop_small_and_large_mol(data, min_atoms=2, max_atoms=128, smiles_field='smiles', addH=True):
    num_atoms_list = []
    for smiles in data[smiles_field]:
        mol = Chem.MolFromSmiles(smiles)
        if addH:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)
        num_atoms = mol.GetNumAtoms()
        num_atoms_list.append(num_atoms)
    
    return data.loc[(np.array(num_atoms_list) >= min_atoms) & (np.array(num_atoms_list) <= max_atoms), :]


class Graph_Bert_Dataset(object):
    def __init__(self,path,smiles_field='Smiles',addH=True, max_len = 100):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        return x, adjoin_matrix, y, weight

    def tf_numerical_smiles(self, data):

        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])

        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight


class Graph_Tune_Parameter_Dataset_Fusion_DA(object):
    def __init__(self, train_path, val_path, test_path, smiles_field='smiles',label_field='Label',addH=True):
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)

        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        # self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH

    def get_data(self, batch_size, shuffle_val=False, drop_remainder_val=False):
        train_data = drop_small_and_large_mol(self.df_train, min_atoms=2, max_atoms=128, smiles_field=self.smiles_field, addH=self.addH)
        val_data = drop_small_and_large_mol(self.df_val, min_atoms=2, max_atoms=128, smiles_field=self.smiles_field, addH=self.addH)
        test_data = drop_small_and_large_mol(self.df_test, min_atoms=2, max_atoms=128, smiles_field=self.smiles_field, addH=self.addH)
        
        print("train-data-len:", len(train_data))
        print("val-data-len:", len(val_data))
        print("test-data-len:", len(test_data))

        self.dataset1 = tf.data.Dataset.from_tensor_slices((
            train_data[self.smiles_field], 
            train_data[self.label_field], 
            train_data.iloc[:, 2:842].values, 
            train_data.iloc[:, 842:].values
        ))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache()
        self.dataset1 = self.dataset1.shuffle(buffer_size=len(train_data))

        self.dataset1 = self.dataset1.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]), 
                tf.TensorShape([None, None]), 
                tf.TensorShape([1]), 
                tf.TensorShape([840]), 
                tf.TensorShape([train_data.shape[1]-842])
            ),
            drop_remainder=True
        ).prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((
            val_data[self.smiles_field], 
            val_data[self.label_field], 
            val_data.iloc[:, 2:842].values, 
            val_data.iloc[:, 842:].values
        ))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).cache()
        
        if shuffle_val:
            self.dataset3 = self.dataset3.shuffle(buffer_size=len(val_data))
        
        self.dataset3 = self.dataset3.padded_batch(
            batch_size,
            padded_shapes=(
                tf.TensorShape([None]), 
                tf.TensorShape([None, None]), 
                tf.TensorShape([1]), 
                tf.TensorShape([840]), 
                tf.TensorShape([val_data.shape[1]-842])
            ),
            drop_remainder=drop_remainder_val
        ).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((
            test_data[self.smiles_field], 
            test_data[self.label_field], 
            test_data.iloc[:, 2:842].values, 
            test_data.iloc[:, 842:].values
        ))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(
            512,
            padded_shapes=(
                tf.TensorShape([None]), 
                tf.TensorShape([None, None]), 
                tf.TensorShape([1]), 
                tf.TensorShape([840]), 
                tf.TensorShape([test_data.shape[1]-842])
            ),
            drop_remainder=False
        ).cache().prefetch(100)


        return self.dataset1, self.dataset2, self.dataset3

    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('int64')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label, bioassay_features, sa_features):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
        x_bioassay = tf.cast(tf.convert_to_tensor(bioassay_features), tf.float32)
        x_sa = tf.cast(tf.convert_to_tensor(sa_features), tf.float32)
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        x_bioassay.set_shape([840])
        x_sa.set_shape([sa_features.shape[0]])
        return x, adjoin_matrix , y, x_bioassay, x_sa

    def __init__(self,path,smiles_field='Smiles',label_field='Label',normalize=True,max_len=100,addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len()<=max_len]
        self.addH =  addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min)-0.5
            self.value_range = self.max-self.min


    def get_data(self):
        data = self.df
        data = data.dropna()

        train_idx = data.sample(frac=0.9).index
        train_data = data[data.index.isin(train_idx)]
        test_data = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(256, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]))).shuffle(100).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1,self.dataset2

    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y
