import re
import pandas as pd

def cleaner(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    code = re.sub('\n', '', code)
    code = re.sub('\t', '', code)
    return code

def load_data(dataset):
    if dataset == "devign":
        train_index, valid_index, test_index = set(), set(), set()
        with open('data/finetune/devign/train.txt') as f:
            train_index.update(int(line.strip()) for line in f)
        with open('data/finetune/devign/valid.txt') as f:
            valid_index.update(int(line.strip()) for line in f)
        with open('data/finetune/devign/test.txt') as f:
            test_index.update(int(line.strip()) for line in f)

        input_dataset = pd.read_json('data/finetune/devign/Devign.json')
        m1, m2, m3 = input_dataset.iloc[list(train_index)], input_dataset.iloc[list(valid_index)], input_dataset.iloc[list(test_index)]
        for df in [m1, m2, m3]:
            df["func"] = df["func"].apply(cleaner)
    else:
        m1 = pd.read_pickle(f'data/finetune/{dataset}/{dataset}_train.pkl')
        m2 = pd.read_pickle(f'data/finetune/{dataset}/{dataset}_val.pkl')
        m3 = pd.read_pickle(f'data/finetune/{dataset}/{dataset}_test.pkl')
        for df in [m1, m2, m3]:
            if "functionSource" in df.columns:
                df["func"] = df["functionSource"].apply(cleaner)
                
            if dataset == "draper":
                df["target"] = df["combine"] * 1

            if "label" in df.columns:
                df["target"] = df["label"]

            if dataset == "mvd":
                df["target"] = df["target"].apply(lambda x: 1 if x != 0 else 0)

    m1 = m1[["func", "target"]]
    m2 = m2[["func", "target"]]
    m3 = m3[["func", "target"]]

    return m1, m2, m3

for dataset in ["devign", "reveal", "vuldeepecker", "mvd", "draper"]:

    print(f"start loading metadata {dataset}")
    m1, m2, m3 = load_data(dataset)

    print(f"In target: training set [1: {len(m1[m1['target']==1])}], 0: [{len(m1[m1['target']==0])}], 1/0 raio: [{len(m1[m1['target']==1])/len(m1[m1['target']==0])}]")
    print(f"In target: validing set [1: {len(m2[m2['target']==1])}], 0: [{len(m2[m2['target']==0])}], 1/0 raio: [{len(m2[m2['target']==1])/len(m2[m2['target']==0])}]")
    print(f"In target: testting set [1: {len(m3[m3['target']==1])}], 0: [{len(m3[m3['target']==0])}], 1/0 raio: [{len(m3[m3['target']==1])/len(m3[m3['target']==0])}]")

    print("*"*28)