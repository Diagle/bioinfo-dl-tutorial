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
    加载一个示例 DNA 数据集（教学用），结构类似 sklearn.load_iris()

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