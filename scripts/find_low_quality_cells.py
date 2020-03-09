import numpy as np
import scanpy as sc
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
import scanpy as sc
import pandas as pd
import glob
from tqdm import tqdm, trange
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

data_dict = {}
gene_list = []
for f in tqdm(glob.glob("/home/scai/processing_simoncai/10xfreshfrozen/data/adata/no_filter/*")):
    data = sc.read_h5ad(f)
    f = f.replace('/home/scai/processing_simoncai/10xfreshfrozen/data/adata/no_filter/','').replace('.h5ad','')
    data_dict[f] = data
    gene_list = data.var_names
threshold_dict = {'D1X_FA2_FRE': 4000,
                 'D1X_FA2_THA': 3500,
                 'D1X_FA3_FRE': 6000,
                 'D1A_FA3_THA': 9000,
                 'D4X_FA2_FRE': 5000,
                 'D4X_FA2_THA': 4000,
                 'D4X_FA3_FRE': 4000,
                 'D4X_FA3_THA': 4000}

combined = {}
for sample in tqdm(sorted(data_dict)):
    sample_name = '_'.join(sample.split('_')[:3])
    tmp = data_dict[sample].to_df()
    if sample_name not in threshold_dict.keys():
        continue
    tmp['low_quality'] = tmp.apply(
        lambda row: True if np.sum(row) < threshold_dict[sample_name] else False,
        axis=1)
    tmp['sample'] = sample_name
    combined[sample_name] = tmp

combined_df = pd.concat(combined)

combined_df['sample'] = combined_df['sample'].str.replace('D1A','D1X')
combined_df = combined_df.reset_index(drop=True)

columns = [
    'val_AUC_freshonly',
    'val_Accuracy_freshonly',
    'val_F1_freshonly',
    'all_tha_AUC_freshonly',
    'all_tha_Accuracy_freshonly',
    'all_tha_F1_freshonly',
    'holdout_fre_AUC_freshonly',
    'holdout_fre_Accuracy_freshonly',
    'holdout_fre_F1_freshonly',
    'holdout_tha_AUC_freshonly',
    'holdout_tha_Accuracy_freshonly',
    'holdout_tha_F1_freshonly',
    'val_AUC_frozenonly',
    'val_Accuracy_frozenonly',
    'val_F1_frozenonly',
    'all_fre_AUC_frozenonly',
    'all_fre_Accuracy_frozenonly',
    'all_fre_F1_frozenonly',
    'holdout_fre_AUC_frozenonly',
    'holdout_fre_Accuracy_frozenonly',
    'holdout_fre_F1_frozenonly',
    'holdout_tha_AUC_frozenonly',
    'holdout_tha_Accuracy_frozenonly',
    'holdout_tha_F1_frozenonly',
    'val_AUC_both',
    'val_Accuracy_both',
    'val_F1_both',
    'holdout_fre_AUC_both',
    'holdout_fre_Accuracy_both',
    'holdout_fre_F1_both',
    'holdout_tha_AUC_both',
    'holdout_tha_Accuracy_both',
    'holdout_tha_F1_both'
]

for i in trange(100):
    for hold_out_sample in tqdm(['D1X_FA2', 'D1X_FA3', 'D4X_FA2', 'D4X_FA3']):
        res = {}
        for col in columns:
            res[col] = []
        combined_holdout_df = combined_df[combined_df['sample'].str.startswith(hold_out_sample)]
        combined_train_df = combined_df[~combined_df['sample'].str.startswith(hold_out_sample)]
        combined_train_fre_df = combined_train_df[combined_train_df['sample'].str.endswith('FRE')]
        combined_train_tha_df = combined_train_df[combined_train_df['sample'].str.endswith('THA')]
        combined_holdout_fre_df = combined_holdout_df[combined_holdout_df['sample'].str.endswith('FRE')]
        combined_holdout_tha_df = combined_holdout_df[combined_holdout_df['sample'].str.endswith('THA')]
        
        # only train fresh
        X = combined_train_fre_df.drop(['low_quality', 'sample'], axis=1)
        y = combined_train_fre_df['low_quality']
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        shuffle=True,
                                                        test_size=0.1,
                                                        stratify=y)
        X_train = StandardScaler().fit_transform(X_train)
        clf = LGBMClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X_train, y_train)
        X_test = StandardScaler().fit_transform(X_test)
        y_conf = clf.predict_proba(X_test)[:,1]
        res['val_AUC_freshonly'].append(roc_auc_score(y_test, y_conf))
        res['val_Accuracy_freshonly'].append(accuracy_score(y_test, (y_conf>0.5)))
        res['val_F1_freshonly'].append(f1_score(y_test, (y_conf>0.5)))
        
        X_tha = combined_train_tha_df.drop(['low_quality', 'sample'], axis=1)
        X_tha = StandardScaler().fit_transform(X_tha)
        y_tha = combined_train_tha_df['low_quality']
        y_tha_conf = clf.predict_proba(X_tha)[:,1]
        res['all_tha_AUC_freshonly'].append(roc_auc_score(y_tha, y_tha_conf))
        res['all_tha_Accuracy_freshonly'].append(accuracy_score(y_tha, (y_tha_conf>0.5)))
        res['all_tha_F1_freshonly'].append(f1_score(y_tha, (y_tha_conf>0.5)))
        
        X_fre_holdout = combined_holdout_fre_df.drop(['low_quality', 'sample'], axis=1)
        y_fre_holdout = combined_holdout_fre_df['low_quality']
        X_fre_holdout = StandardScaler().fit_transform(X_fre_holdout)
        y_fre_holdout_conf = clf.predict_proba(X_fre_holdout)[:,1]
        res['holdout_fre_AUC_freshonly'].append(roc_auc_score(y_fre_holdout, y_fre_holdout_conf))
        res['holdout_fre_Accuracy_freshonly'].append(accuracy_score(y_fre_holdout, (y_fre_holdout_conf>0.5)))
        res['holdout_fre_F1_freshonly'].append(f1_score(y_fre_holdout, (y_fre_holdout_conf>0.5)))
        
        X_tha_holdout = combined_holdout_tha_df.drop(['low_quality', 'sample'], axis=1)
        y_tha_holdout = combined_holdout_tha_df['low_quality']
        X_tha_holdout = StandardScaler().fit_transform(X_tha_holdout)
        y_tha_holdout_conf = clf.predict_proba(X_tha_holdout)[:,1]
        res['holdout_tha_AUC_freshonly'].append(roc_auc_score(y_tha_holdout, y_tha_holdout_conf))
        res['holdout_tha_Accuracy_freshonly'].append(accuracy_score(y_tha_holdout, (y_tha_holdout_conf>0.5)))
        res['holdout_tha_F1_freshonly'].append(f1_score(y_tha_holdout, (y_tha_holdout_conf>0.5)))
        
        # only train frozen
        X = combined_train_tha_df.drop(['low_quality', 'sample'], axis=1)
        y = combined_train_tha_df['low_quality']
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        shuffle=True,
                                                        test_size=0.1,
                                                        stratify=y)
        X_train = StandardScaler().fit_transform(X_train)
        clf = LGBMClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X_train, y_train)
        X_test = StandardScaler().fit_transform(X_test)
        y_conf = clf.predict_proba(X_test)[:,1]
        res['val_AUC_frozenonly'].append(roc_auc_score(y_test, y_conf))
        res['val_Accuracy_frozenonly'].append(accuracy_score(y_test, (y_conf>0.5)))
        res['val_F1_frozenonly'].append(f1_score(y_test, (y_conf>0.5)))
        
        X_fre = combined_train_fre_df.drop(['low_quality', 'sample'], axis=1)
        X_fre = StandardScaler().fit_transform(X_fre)
        y_fre = combined_train_fre_df['low_quality']
        y_fre_conf = clf.predict_proba(X_fre)[:,1]
        res['all_fre_AUC_frozenonly'].append(roc_auc_score(y_fre, y_fre_conf))
        res['all_fre_Accuracy_frozenonly'].append(accuracy_score(y_fre, (y_fre_conf>0.5)))
        res['all_fre_F1_frozenonly'].append(f1_score(y_fre, (y_fre_conf>0.5)))
        
        X_fre_holdout = combined_holdout_fre_df.drop(['low_quality', 'sample'], axis=1)
        y_fre_holdout = combined_holdout_fre_df['low_quality']
        X_fre_holdout = StandardScaler().fit_transform(X_fre_holdout)
        y_fre_holdout_conf = clf.predict_proba(X_fre_holdout)[:,1]
        res['holdout_fre_AUC_frozenonly'].append(roc_auc_score(y_fre_holdout, y_fre_holdout_conf))
        res['holdout_fre_Accuracy_frozenonly'].append(accuracy_score(y_fre_holdout, (y_fre_holdout_conf>0.5)))
        res['holdout_fre_F1_frozenonly'].append(f1_score(y_fre_holdout, (y_fre_holdout_conf>0.5)))
        
        X_tha_holdout = combined_holdout_tha_df.drop(['low_quality', 'sample'], axis=1)
        y_tha_holdout = combined_holdout_tha_df['low_quality']
        X_tha_holdout = StandardScaler().fit_transform(X_tha_holdout)
        y_tha_holdout_conf = clf.predict_proba(X_tha_holdout)[:,1]
        res['holdout_tha_AUC_frozenonly'].append(roc_auc_score(y_tha_holdout, y_tha_holdout_conf))
        res['holdout_tha_Accuracy_frozenonly'].append(accuracy_score(y_tha_holdout, (y_tha_holdout_conf>0.5)))
        res['holdout_tha_F1_frozenonly'].append(f1_score(y_tha_holdout, (y_tha_holdout_conf>0.5)))
        
        # both
        X = combined_train_df.drop(['low_quality', 'sample'], axis=1)
        y = combined_train_df['low_quality']
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        shuffle=True,
                                                        test_size=0.1,
                                                        stratify=y)
        X_train = StandardScaler().fit_transform(X_train)
        clf = LGBMClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X_train, y_train)
        X_test = StandardScaler().fit_transform(X_test)
        y_conf = clf.predict_proba(X_test)[:,1]
        res['val_AUC_both'].append(roc_auc_score(y_test, y_conf))
        res['val_Accuracy_both'].append(accuracy_score(y_test, (y_conf>0.5)))
        res['val_F1_both'].append(f1_score(y_test, (y_conf>0.5)))
        
        X_fre_holdout = combined_holdout_fre_df.drop(['low_quality', 'sample'], axis=1)
        y_fre_holdout = combined_holdout_fre_df['low_quality']
        X_fre_holdout = StandardScaler().fit_transform(X_fre_holdout)
        y_fre_holdout_conf = clf.predict_proba(X_fre_holdout)[:,1]
        res['holdout_fre_AUC_both'].append(roc_auc_score(y_fre_holdout, y_fre_holdout_conf))
        res['holdout_fre_Accuracy_both'].append(accuracy_score(y_fre_holdout, (y_fre_holdout_conf>0.5)))
        res['holdout_fre_F1_both'].append(f1_score(y_fre_holdout, (y_fre_holdout_conf>0.5)))
        
        X_tha_holdout = combined_holdout_tha_df.drop(['low_quality', 'sample'], axis=1)
        y_tha_holdout = combined_holdout_tha_df['low_quality']
        X_tha_holdout = StandardScaler().fit_transform(X_tha_holdout)
        y_tha_holdout_conf = clf.predict_proba(X_tha_holdout)[:,1]
        res['holdout_tha_AUC_both'].append(roc_auc_score(y_tha_holdout, y_tha_holdout_conf))
        res['holdout_tha_Accuracy_both'].append(accuracy_score(y_tha_holdout, (y_tha_holdout_conf>0.5)))
        res['holdout_tha_F1_both'].append(f1_score(y_tha_holdout, (y_tha_holdout_conf>0.5)))
        res_all = pd.read_csv("../results/threshold_ml.csv")
        res_all = pd.concat([res_all, pd.DataFrame(res)])
        res_all.to_csv("../results/threshold_ml.csv", index=False)

