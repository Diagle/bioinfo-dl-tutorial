import os
import pandas as pd
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

def load_bulk_drug_response():
    """
    Returns
    -------
    Bunch
        data:基因表达信息(pandas)\n
        target:细胞系耐药结果(sensitive/resisted)\n
        feature_names:(gene)\n
        target_names:(敏感1/耐药0)\n
        DESCR:"A cellliens cancer drug response dataset."
    """
    data_path = os.path.join(DATA_DIR, "drug_response", "Erlotinib_data.txt")
    label_path = os.path.join(DATA_DIR, "drug_response", "Erlotinib_label.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset file: {data_path}")
    
    data = pd.read_csv(data_path,sep='\t',index_col=0).T
    target = pd.read_csv(label_path,sep='\t',index_col=0)
    target_names = sorted(list(set(target)))
    return Bunch(
        data=data,
        target=target,
        feature_names=["gene"],
        target_names=target_names,
        DESCR="A cellliens cancer drug response dataset."
    )

def load_sc_drug_response():
    """
    Returns
    -------
    Bunch
        data:基因表达信息(pandas)\n
        target:细胞系耐药结果(sensitive/resisted)\n
        feature_names:(gene)\n
        target_names:(敏感1/耐药0)\n
        DESCR:"A cellliens cancer drug response dataset."
    """


def load_cancer():
    """
    Returns
    -------
    Bunch
        data:基因表达信息(pandas)\n
        target:DFS时间,DFS事件(months,1/0)\n
        feature_names:(gene)\n
        target_names:(1/耐药0)\n
        DESCR:"A cellliens cancer drug response dataset."
    """
    data_path = os.path.join(DATA_DIR, "drug_response", "Erlotinib_data.txt")
    label_path = os.path.join(DATA_DIR, "drug_response", "Erlotinib_label.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset file: {data_path}")
    
    data = pd.read_csv(data_path,sep='\t',index_col=0).T
    target = pd.read_csv(label_path,sep='\t',index_col=0)
    target_names = sorted(list(set(target)))
    return Bunch(
        data=data,
        target=target,
        feature_names=["gene"],
        target_names=target_names,
        DESCR="A cellliens cancer drug response dataset."
    )

    
if __name__ == "__main__":
    drug_response =  load_cellines_drug_response()
    data = drug_response.data
    label = drug_response.target
else:
    print("data load successfully!")

