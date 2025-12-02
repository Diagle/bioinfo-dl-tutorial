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

def load_tiny_dna():
    """
    Returns
    -------
    Bunch
        data: DNA 序列（str）
        target: 标签（如 enhancer / promoter）
        target_names: 类别名称
        DESCR: 描述
    """
    path = os.path.join(DATA_DIR, "example", "tiny_dna.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find dataset file: {path}")

    df = pd.read_csv(path)

    data = df["sequence"].tolist()
    target = df["label"].tolist()
    target_names = sorted(list(set(target)))

    return Bunch(
        data=data,
        target=target,
        feature_names=["sequence"],
        target_names=target_names,
        DESCR="A tiny DNA sequence dataset for demonstration."
    )

def load_cellines_drug_response():
    """
    Returns
    -------
    Bunch
        data:基因表达信息(pandas)
        target:细胞系耐药结果(sensitive/resisted)
        feature_names:(gene)
        target_names:(敏感1/耐药0)
        DESCR:"A cellliens cancer drug response dataset."
    """
    data_path = os.path.join(DATA_DIR, "drug_resonse", "Erlotinib_data.txt")
    label_path = os.path.join(DATA_DIR, "drug_resonse", "Erlotinib_label.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find dataset file: {data_path}")
    
    data = pd.read_csv(data_path,sep='\t',index_col=0).T
    target = pd.read_csv('./Cetuximab_label.txt',sep='\t',index_col=0)['Class']
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

else:
    print("data load successfully!")