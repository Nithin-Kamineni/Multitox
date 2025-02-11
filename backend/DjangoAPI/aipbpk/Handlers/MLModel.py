from keras.models import load_model
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import load_model

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
import sqlite3
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np
import pandas as pd
from keras.models import load_model
import tensorflow as tf

# Import necessary packages
from sklearn.preprocessing import LabelEncoder

# Data preprocessing for numerical data type
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# One-hot encoding for categorical data type
from sklearn.preprocessing import OneHotEncoder

import pubchempy as pcp
from pubchempy import get_compounds, Compound
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs, MACCSkeys, Descriptors   # Molecular descriptors and fingerprint generation
from rdkit.DataStructs import ExplicitBitVect          # Data structures for working with molecular descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors   # Calculation of molecular descriptors
from sklearn.preprocessing import MinMaxScaler        # Feature scaling for RDKit descriptors

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2

def cas_to_cid(df):
    df['PubChemCID'] = None
    df['Attempts'] = 0  # Add a column to track the number of attempts

    # Searching PubChem for CID
    for i, CAS in df['CAS'].items():
        try:
            # Searching PubChem for CID using CAS number
            results = get_compounds(CAS, 'name')
            if results:
                cid = results[0].cid
                df.at[i, 'PubChemCID'] = cid
            else:
                print(f"No results found for CAS number {CAS}")

        except Exception as e:
            pass  # Silent the error message; otherwise, it will print a lot of error messages

    # Check PubChemCID; If CID is None, run the loop again with a maximum of 3 attempts
    while df['PubChemCID'].isnull().sum() > 0:
        #print(f"Remaining null CIDs: {df['PubChemCID'].isnull().sum()}")

        for i, CAS in df[df['PubChemCID'].isnull()]['CAS'].items():
            if df.at[i, 'Attempts'] >= 3:
                #print(f"Max attempts reached for CAS number {CAS}. Marking as False.")
                df.at[i, 'PubChemCID'] = False  # Mark as False if CID not found after 3 attempts
                continue

            try:
                # Increment the attempt counter
                df.at[i, 'Attempts'] += 1

                # Searching PubChem for CID using CAS number
                results = get_compounds(CAS, 'name')
                if results:
                    cid = results[0].cid
                    df.at[i, 'PubChemCID'] = cid
                else:
                    print(f"No results found for CAS number {CAS} on attempt {df.at[i, 'Attempts']}")

            except Exception as e:
                pass

        # Break the loop if no more null values are left or all attempts are completed
        if (df['Attempts'] >= 3).all():
            break

    # Drop the 'Attempts' column if not needed in the final dataframe
    df.drop(columns=['Attempts'], inplace=True)

    return df

# Function for searching and extracting SMILES code with entering CID
def cid_to_smiles (data):
  data['SMILES'] = None

  for i, cid in data['PubChemCID'].items():
    try:
      compound = pcp.Compound.from_cid(cid)
      if compound:
        smiles = compound.canonical_smiles
        data.at[i, 'SMILES'] = smiles
      else:
        print(f'No results found for PubChemCID {cid}')

    except Exception as e: # a general exception handler
        pass # If an exception is caught, it will ignored, and the code will proceed without raising an error to avoid #printing a lot of error messages related to a specific situation ("PUG-REST server is busy").

  # Check SMILES; If SMILES is None, run the "While loop" and request PubChem server again to get all SMILES
  while data['SMILES'].isnull().sum() > 0:
    #print (data['SMILES'].isnull().sum())
    for i, cid in data[data['SMILES'].isnull()]['PubChemCID'].items():
     try:
       compound = pcp.Compound.from_cid(cid)
       if compound:
        smiles = compound.canonical_smiles
        data.at[i, 'SMILES'] = smiles
       else:
        print(f'No results found for PubChemCID {cid}')
     except Exception as e:
       pass

  return data

# Define a function that transforms SMILES string into RDKIT descriptors
def cal_rdkit_descr(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    mol_descriptors = []
    for mol in mols:
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)

    return pd.DataFrame(mol_descriptors, columns=["rdkit_" + str(i) for i in desc_names])


# Define a function that transforms a SMILES string into an FCFP (if use_features = TRUE) or--
# --the Extended-Connectivity Finger#prints (ECFP) descriptors (if use_features = FALSE)

def cal_ECFP6_descr(smiles,
            R = 3,
            nBits = 2**10, # nBits = 1024
            use_features = False,
            use_chirality = False):

   '''
   Inputs:
   - smiles...SMILES string of input compounds
   - R....Maximum radius of circular substructures--By using this radius parameter, we compute ECFP6 (the equivalent of radius 3)
   - nBits....number of bits, default is 2048. 1024 is also widely used.
   - use_features...if true then use pharmacophoric atom features (FCFPs), if false then use standard DAYLIGHT atom features (ECFP)
   - use_chirality...if true then append tetrahedral chirality flags to atom features
   Outputs:
   - pd.DataFrame...ECFP or FCFPs with length nBits and maximum radus R

   '''
   mols = [AllChem.MolFromSmiles(i) for i in smiles]

   ecfp_descriptors = []
   for mol in mols:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                radius = R,
                                nBits = nBits,
                                useFeatures = use_features,
                                useChirality = use_chirality)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(ecfp, array)
        ecfp_descriptors.append(ecfp)

   return pd.DataFrame([list(l) for l in ecfp_descriptors], columns=[f'ECFP6_Bit_{i}' for i in range(nBits)])


# Define a function that transforms a SMILES string into an FCFP (if use_features = TRUE)
def cal_FCFP6_descr(smiles,
            R = 3,
            nBits = 2**10, # nBits = 1024
            use_features = True,
            use_chirality = False):

   mols = [AllChem.MolFromSmiles(i) for i in smiles]

   fcfp_descriptors = []
   for mol in mols:
        fcfp = AllChem.GetMorganFingerprintAsBitVect(mol,
                                radius = R,
                                nBits = nBits,
                                useFeatures = use_features,
                                useChirality = use_chirality)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fcfp, array)
        fcfp_descriptors.append(fcfp)

   return pd.DataFrame([list(l) for l in fcfp_descriptors], columns=[f'FCFP6_Bit_{i}' for i in range(nBits)])


# Define a function that transforms a SMILES string into an MACCS finger#prints

def cal_MACCS_descr(smiles):

   mols = [Chem.MolFromSmiles(i) for i in smiles]
   MACCS_descriptors = []
   for mol in mols:
        fp = MACCSkeys.GenMACCSKeys (mol)
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        MACCS_descriptors.append(fp)

   return pd.DataFrame([list(l) for l in MACCS_descriptors], columns=[f'MACCS_Bit_{i}' for i in range(167)])

def search_record(dataframe, cas=None, smiles=None, name=None):
    if cas:
        result = dataframe[dataframe['CAS_x'] == cas]
    elif smiles:
        result = dataframe[dataframe['SMILES'] == smiles]
    elif name:
        result = dataframe[dataframe['Name'] == name]
    else:
        return False
    
    if result.empty:
        return False
    else:
        return result.index.tolist()

class MLModel:
    def __init__(self):
        self.in_vivo_data = pd.read_csv('static/hdata_Hu_PW2_0712.csv')
        in_vivo_data_tr = self.in_vivo_data[~(self.in_vivo_data['SMILES'].isnull())].reset_index(drop=True)

        self.tox21_data = pd.read_csv('static/tox21_all-clean.csv')
        self.y = in_vivo_data_tr [in_vivo_data_tr['CAS'].isin(self.tox21_data['CAS'])].iloc[:,4:10].reset_index(drop=True)

        self.input_df = pd.merge(self.in_vivo_data.iloc[:,[0,1,2]], self.tox21_data, on = 'CAS', how='inner').reset_index(drop=True)
        self.input_df = self.input_df[~(self.input_df['SMILES'].isnull())] .iloc[:,[0,1,2]]

        self.X_CAS = self.in_vivo_data.iloc[:,[0,2]] # reference CAS

        X_tox21 = pd.merge(self.X_CAS, self.tox21_data, on = 'CAS', how='inner').reset_index(drop=True)
        X_tox21 = X_tox21[~(X_tox21['SMILES'].isnull())]

        self.combined_df = pd.merge(in_vivo_data_tr, X_tox21, on='SMILES', how='inner')

        rdkit_descrs = cal_rdkit_descr(smiles=X_tox21['SMILES'])

        missing_data = rdkit_descrs.isnull().sum()
        missing_columns = missing_data[missing_data > 0]

        rdkit_descrs_clean = rdkit_descrs.drop(columns=missing_columns.index)
        scaler = MinMaxScaler()
        X_rdkit_descrs = rdkit_descrs_clean
        X_rdkit = scaler.fit_transform(X_rdkit_descrs)
        X_rdkit = pd.DataFrame(X_rdkit, columns = rdkit_descrs_clean.columns.values.tolist())
        X_rdkit.insert(0, 'CAS', X_tox21['CAS'].values, False)

        X_rdkit_NCAS = X_rdkit.drop(columns=['CAS'])

        X_ECFP = cal_ECFP6_descr(smiles=X_tox21['SMILES'])
        X_ECFP.insert(0, 'CAS', X_tox21['CAS'].values, False)

        X_ECFP_NCAS = X_ECFP.drop(columns=['CAS'])

        X_FCFP = cal_FCFP6_descr(smiles=X_tox21['SMILES'])
        X_FCFP.insert(0, 'CAS', X_tox21['CAS'].values, False)

        X_FCFP_NCAS = X_FCFP.drop(columns=['CAS'])

        X_MACCS = cal_MACCS_descr(smiles=X_tox21['SMILES'])
        X_MACCS.insert(0, 'CAS', X_tox21['CAS'].values, False)

        X_MACCS_NCAS = X_MACCS.drop(columns=['CAS'])

        X_tox21_tr = X_tox21.drop(['SMILES','ID'], axis=1)

        X_tox21_tr_NCAS = X_tox21_tr.drop(columns=['CAS'])

        X_rdkit_tox21 = pd.merge(X_rdkit, X_tox21_tr, on = 'CAS', how='inner')

        X_rdkit_tox21_NCAS = X_rdkit_tox21.drop(columns=['CAS'])

        X_ECFP_tox21 = pd.merge(X_ECFP, X_tox21_tr, on = 'CAS', how='inner')

        X_ECFP_tox21_NCAS = X_ECFP_tox21.drop(columns=['CAS'])

        X_FCFP_tox21 = pd.merge(X_FCFP, X_tox21_tr, on = 'CAS', how='inner')

        X_FCFP_tox21_NCAS = X_FCFP_tox21.drop(columns=['CAS'])

        X_MACCS_tox21 = pd.merge(X_MACCS, X_tox21_tr, on = 'CAS', how='inner')

        X_MACCS_tox21_NCAS = X_MACCS_tox21.drop(columns=['CAS'])

        X_all_tox21 = pd.merge(X_rdkit, X_ECFP, on='CAS', how='inner')\
                .merge(X_FCFP, on='CAS', how='inner')\
                .merge(X_MACCS, on='CAS', how='inner')\
                .merge(X_tox21_tr, on='CAS', how='inner')

        X_all_tox21_NCAS = X_all_tox21.drop(columns=['CAS'])

        X_all = pd.merge(X_rdkit, X_ECFP, on='CAS', how='inner')\
                .merge(X_FCFP, on='CAS', how='inner')\
                .merge(X_MACCS, on='CAS', how='inner')

        X_all_NCAS = X_all.drop(columns=['CAS'])

        model_save_path = f"static/best_model_neuro_not_include.h5"
        self.overall_best_model_nni = tf.keras.models.load_model(model_save_path)
        #print(f"Model loaded from {model_save_path}")

        model_save_path = f"static/best_model_neuro_include.h5"
        self.overall_best_model_ni = tf.keras.models.load_model(model_save_path)
        # from tensorflow.keras.models import load_model
        # overall_best_model = load_model(model_path)
        #print(f"Model loaded from {model_save_path}")

        feature_sets = [X_MACCS_tox21_NCAS]
        feature_names = ['X_MACCS_tox21']
        outer_cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for feature_set, feature_name in zip(feature_sets, feature_names):
            for fold_num, (outer_train_ix, outer_test_ix) in enumerate(outer_cv.split(feature_set, self.y), start=1):
                X_test_outer = feature_set.iloc[outer_test_ix]
                # X_test_outer_selected = select_feature.transform(X_test_outer)
            # break

        # Define a threshold for classification
        threshold = 0.5

        # Predictions from the best-performing model
        # y_hat = overall_best_model.predict(X_test_outer_selected)  # Ensure X_test_outer_selected is defined
        y_hat = self.overall_best_model_ni.predict(feature_set)  # Ensure X_test_outer_selected is defined

        # Ensure `toxicity_labels` matches the column names of `y`
        toxicity_labels = self.y.columns.tolist()  # Extract toxicity labels from `y`

        # Function to classify a chemical based on predictions
        def classify_toxicity(y_pred_row):
            return ['Y' if prob >= threshold else 'N' for prob in y_pred_row]

        # Apply the classification function to all predictions
        classification_results = np.apply_along_axis(classify_toxicity, 1, y_hat)

        # Create a DataFrame to store the classification results
        self.classification_df_ni = pd.DataFrame(classification_results, columns=toxicity_labels)

        model_save_path = f"static/best_model_rem_not_include.h5"
        self.overall_best_model_rni = tf.keras.models.load_model(model_save_path)
        # from tensorflow.keras.models import load_model
        # overall_best_model = load_model(model_path)
        #print(f"Model loaded from {model_save_path}")

        model_save_path = f"static/best_model_rem_include.h5"
        self.overall_best_model_rin = tf.keras.models.load_model(model_save_path)
        # from tensorflow.keras.models import load_model
        # overall_best_model = load_model(model_path)
        #print(f"Model loaded from {model_save_path}")

        feature_sets = [X_all_tox21_NCAS]
        feature_names = ['X_all_tox21']

        outer_cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_cv = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for feature_set, feature_name in zip(feature_sets, feature_names):
            #print(f"Running nested cross-validation for feature set: {feature_name}")
            
            for fold_num, (outer_train_ix, outer_test_ix) in enumerate(outer_cv.split(feature_set, self.y), start=1):
                X_train_outer, X_test_outer = feature_set.iloc[outer_train_ix], feature_set.iloc[outer_test_ix]
                y_train_outer, y_test_outer = self.y.iloc[outer_train_ix], self.y.iloc[outer_test_ix]
                
                # Inner loop for hyperparameter tuning and feature selection
                for inner_train_ix, inner_test_ix in inner_cv.split(X_train_outer, y_train_outer):
                    X_train_inner, X_test_inner = X_train_outer.iloc[inner_train_ix], X_train_outer.iloc[inner_test_ix]
                    y_train_inner, y_test_inner = y_train_outer.iloc[inner_train_ix], y_train_outer.iloc[inner_test_ix]
                
                    # select_feature = SelectKBest(chi2, k='all')  # Select all features (no feature selection)
                    select_feature = SelectKBest(chi2, k=2482)  # Select all features (no feature selection)
                    X_train_inner_selected = select_feature.fit_transform(X_train_inner, y_train_inner)
                    X_test_inner_selected = select_feature.transform(X_test_inner)

                # Apply feature selection to outer fold data
                X_test_outer_selected = select_feature.transform(feature_set)

        # Define a threshold for classification
        threshold = 0.5

        # Predictions from the best-performing model
        y_hat = self.overall_best_model_rin.predict(X_test_outer_selected)  # Ensure X_test_outer_selected is defined

        # Ensure `toxicity_labels` matches the column names of `y`
        toxicity_labels = self.y.columns.tolist()  # Extract toxicity labels from `y`

        # Function to classify a chemical based on predictions
        def classify_toxicity(y_pred_row):
            return ['Y' if prob >= threshold else 'N' for prob in y_pred_row]

        # Apply the classification function to all predictions
        classification_results = np.apply_along_axis(classify_toxicity, 1, y_hat)

        # Create a DataFrame to store the classification results
        self.classification_df_ri = pd.DataFrame(classification_results, columns=toxicity_labels)

        # Display the classification results
        #print("\nClassification Results:")
        #print(self.classification_df_ri)

    def GetIndexes(self, SMILES):
        index = search_record(self.combined_df, smiles=SMILES)
        return index

    def GetNeuroNotInluded(self, CASlist):
        threshold = 0.5

        # CASlist=['100-11-8']

        data = {
            'SMILES': [cas for cas in CASlist]
        }

        df_smiles = pd.DataFrame(data)
        # df_cid = cas_to_cid(df)

        # df_smiles = cid_to_smiles(df_cid).drop(columns=['PubChemCID','CAS'])
        
        # Calculating MACCS finger#prints
        test_data_no_rem = cal_MACCS_descr(smiles=df_smiles['SMILES'])
        
        # Predictions from the best-performing model
        # y_hat = overall_best_model.predict(X_test_outer_selected)  # Ensure X_test_outer_selected is defined
        y_hat = self.overall_best_model_nni.predict(test_data_no_rem)  # Ensure X_test_outer_selected is defined
        
        # Ensure `toxicity_labels` matches the column names of `y`
        toxicity_labels = self.y.columns.tolist()  # Extract toxicity labels from `y`
        
        # Function to classify a chemical based on predictions
        def classify_toxicity(y_pred_row):
            return ['Y' if prob >= threshold else 'N' for prob in y_pred_row]
        
        # Apply the classification function to all predictions
        classification_results = np.apply_along_axis(classify_toxicity, 1, y_hat)
        
        # Create a DataFrame to store the classification results
        self.classification_df_rni = pd.DataFrame(classification_results, columns=toxicity_labels)
        
        # Display the classification results
        #print("\nClassification Results:")
        #print(self.classification_df_rni)
        return self.classification_df_rni.to_dict(orient='records')
        
    def GetNeuroInluded(self):
        #print('self.classification_df_ni')
        #print(self.classification_df_ni)
        return self.classification_df_ni.to_dict(orient='records')

    def GetRemNotInluded(self, CASlist):
        # CASlist = ['100-11-8']

        data = {
            'SMILES': [cas for cas in CASlist]
        }

        df_smiles = pd.DataFrame(data)
        # df_cid = cas_to_cid(df)

        #print('---------------------------------------')
        #print(df_cid)

        # df_smiles = cid_to_smiles(df_cid).drop(columns=['PubChemCID', 'CAS'])

        rdkit_descrs_temp = cal_rdkit_descr(smiles=df_smiles['SMILES'])

        #print('---------------------------------------')
        #print('rdkit_descrs_temp',rdkit_descrs_temp)
        missing_data_temp = rdkit_descrs_temp.isnull().sum()
        missing_columns_temp = missing_data_temp[missing_data_temp > 0]
        rdkit_descrs_clean_temp = rdkit_descrs_temp.drop(columns=missing_columns_temp.index)
        scaler = MinMaxScaler()
        X_rdkit_descrs_temp = rdkit_descrs_clean_temp
        X_rdkit_temp = scaler.fit_transform(X_rdkit_descrs_temp)
        X_rdkit_temp = pd.DataFrame(X_rdkit_temp, columns=rdkit_descrs_clean_temp.columns.values.tolist())
        X_rdkit_temp.insert(0, 'CAS', df_smiles['SMILES'].values, False)

        X_ECFP_temp = cal_ECFP6_descr(smiles=df_smiles['SMILES'])
        X_ECFP_temp.insert(0, 'CAS', df_smiles['SMILES'].values, False)

        X_FCFP_temp = cal_FCFP6_descr(smiles=df_smiles['SMILES'])
        X_FCFP_temp.insert(0, 'CAS', df_smiles['SMILES'].values, False)

        X_MACCS_temp = cal_MACCS_descr(smiles=df_smiles['SMILES'])
        X_MACCS_temp.insert(0, 'CAS', df_smiles['SMILES'].values, False)

        #print('---------------------------------------')
        #print('X_rdkit_temp',X_rdkit_temp.shape)
        #print('X_ECFP_temp',X_ECFP_temp.shape)
        #print('X_FCFP_temp',X_FCFP_temp.shape)
        #print('X_MACCS_temp',X_MACCS_temp.shape)

        X_all_temp = pd.merge(X_rdkit_temp, X_ECFP_temp, on='CAS', how='inner')\
                        .merge(X_FCFP_temp, on='CAS', how='inner')\
                        .merge(X_MACCS_temp, on='CAS', how='inner')

        X_all_NCAS_temp = X_all_temp.drop(columns=['CAS'])
        #print('X_all_temp',X_all_temp.shape)

        #print('---------------------------------------X_all_NCAS_temp')
        #print('X_all_NCAS_temp',X_all_NCAS_temp.shape)

        # Align feature selection to expected size
        if X_all_NCAS_temp.shape[1] > 2411:
            X_all_NCAS_temp = X_all_NCAS_temp.iloc[:, :2411]  # Truncate to match expected size
        elif X_all_NCAS_temp.shape[1] < 2411:
            raise ValueError(f"Feature set has fewer features ({X_all_NCAS_temp.shape[1]}) than expected (2411). Please adjust feature selection.")

        # Use the entire dataset for predictions
        feature_set = X_all_NCAS_temp
        #print('---------------------------------------feature_set')
        #print('feature_set',feature_set.shape)

        # Define a threshold for classification
        threshold = 0.5

        # Predictions from the best-performing model
        y_hat = self.overall_best_model_rni.predict(feature_set)

        # Ensure `toxicity_labels` matches the column names of `y`
        toxicity_labels = self.y.columns.tolist()  # Extract toxicity labels from `y`

        # Function to classify a chemical based on predictions
        def classify_toxicity(y_pred_row):
            return ['Y' if prob >= threshold else 'N' for prob in y_pred_row]

        # Apply the classification function to all predictions
        classification_results = np.apply_along_axis(classify_toxicity, 1, y_hat)

        #print("classification_results",classification_results)
        # Create a DataFrame to store the classification results
        classification_df = pd.DataFrame(classification_results, columns=toxicity_labels)

        # Display the classification results
        return classification_df.to_dict(orient='records')

    def GetRemInluded(self):
        return self.classification_df_ri.to_dict(orient='records')

mLModel = MLModel()