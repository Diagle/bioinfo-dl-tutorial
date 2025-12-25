import os
import pandas as pd
import scipy.io as sio

from dataclasses import dataclass


DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

@dataclass
class Bunch:
    """类似 sklearn.utils.Bunch 的结构化数据容器"""
    data: any = None
    target: any = None
    feature_names: list = None
    target_names: list = None
    DESCR: str = ""


def load_amazon_fc6():
    """
    Returns
    -------
    Bunch
        data:ImageNet 第6层深度特征集(pandas)\n
        target:物体类别\n
        feature_names:(深度特征)\n
        target_names:(多类别)\n
        DESCR:"Office-31 amazon ImageNet 第6层 DeCAF 深度学习特征集."
    """
    mat_path = os.path.join(DATA_DIR, "default", "amazon_fc6.mat")

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Cannot find dataset file: {mat_path}")

    mat_data = sio.loadmat(mat_path)
    data = mat_data['fts']
    target = mat_data['labels'].flatten()
    target_names = sorted(list(set(target)))
    return Bunch(
        data=data,
        target=target,
        feature_names=["DeCAF"],
        target_names=target_names,
        DESCR="Office-31 amazon ImageNet 第6层 DeCAF 深度学习特征集。"
    )


def load_dslr_fc6():
    """
    Returns
    -------
    Bunch
        data:ImageNet 第6层深度特征集(pandas)\n
        target:物体类别\n
        feature_names:(深度特征)\n
        target_names:(多类别)\n
        DESCR:"Office-31 DSLR ImageNet 第6层 DeCAF 深度学习特征集."
    """
    mat_path = os.path.join(DATA_DIR, "default", "dslr_fc6.mat")

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Cannot find dataset file: {mat_path}")

    mat_data = sio.loadmat(mat_path)
    data = mat_data['fts']
    target = mat_data['labels'].flatten()
    target_names = sorted(list(set(target)))
    return Bunch(
        data=data,
        target=target,
        feature_names=["DeCAF"],
        target_names=target_names,
        DESCR="Office-31 DSLR ImageNet 第6层 DeCAF 深度学习特征集。"
    )


def load_bulk_drug_response():
    """
    Returns
    -------
    Bunch
        data:bulk rna seq profile\n
        target:bulk cancer drug response\n
        feature_names:genes symbol\n
        target_names:sensitive(1)/resisted(0)\n
        DESCR:"Cellliens cancer drug(Erlotinib) response dataset from GDSC and CCLE."
    """
    data_path = os.path.join(DATA_DIR, "drug_response", "Erlotinib_data.txt")
    label_path = os.path.join(DATA_DIR, "drug_response", "Erlotinib_label.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset file: {data_path}")
    
    data = pd.read_csv(data_path,sep='\t',index_col=0).T
    target = pd.read_csv(label_path,sep='\t',index_col=0).values.flatten()
    target_names = sorted(list(set(target)))
    
    return Bunch(
        data=data,
        target=target,
        feature_names=["genes symbol"],
        target_names=target_names,
        DESCR="Cellliens cancer drug(Erlotinib) response dataset from GDSC and CCLE."
    )

def load_sc_drug_response():
    """
    Returns
    -------
    Bunch
        data:singlecell rna seq profile\n
        target:single cancer drug response\n
        feature_names:genes symbol\n
        target_names:sensitive(1)/resisted(0)\n
        DESCR:"singlecell cancer drug(Erlotinib) response dataset from GSE149383."
    """
    info_path = os.path.join(DATA_DIR, "drug_response", "GSE149383_Erlotinib_counts_response.csv")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Cannot find dataset file: {info_path}")
    
    info = pd.read_csv(info_path,sep=',',index_col=0)
    data = info.iloc[:,1:]
    target = info.iloc[:,0].values.flatten()
    target_names = sorted(list(set(target)))
    
    return Bunch(
        data=data,
        target=target,
        feature_names=["genes symbol"],
        target_names=target_names,
        DESCR="singlecell cancer drug(Erlotinib) response dataset from GSE149383."
    )



def load_cancer():
    """
    Returns
    -------
    Bunch
        data:TCGA patient gene expression matrix\n
        target:DFS (in months)\n
        feature_names:genes probe\n
        target_names:1/0(Disease-Free Survival)\n
        DESCR:"A TCGA patient cancer disease-free survival event and time dataset."

    """
    data_path = os.path.join(DATA_DIR, "cancer", "Colorectal Cancer Gene Expression Data.csv")
    label_path = os.path.join(DATA_DIR, "cancer", "Colorectal Cancer Patient Data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset file: {data_path}")
    
    data = pd.read_csv(data_path,sep=',',index_col=1).iloc[:,1:].T
    target = pd.read_csv(label_path,sep=',',index_col=1).loc[:,'DFS (in months)'].values.flatten()
    target_names = sorted(list(set(target)))

    return Bunch(
        data=data,
        target=target,
        feature_names=["genes probe"],
        target_names=target_names,
        DESCR="A TCGA patient cancer disease-free survival event and time dataset."
    )
    
if __name__ == "__main__":
    df =  load_bulk_drug_response()
    data = df.data
    label = df.target

else:
    print("data load successfully!")

