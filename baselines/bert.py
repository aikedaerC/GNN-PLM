import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel
import re
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from joblib import dump, load
import wandb

def select(result, filter_column_value):
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    return result

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


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (x1, x2, y) in enumerate(train_loader): #, total=len(train_loader), desc=f"Training"):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        y_pred = model(x1, x2)
        loss = criterion(y_pred, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    for batch_idx, (x1, x2, y)  in enumerate(test_loader): #, total=len(test_loader), desc=f"evaluating"):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x1, x2)
        test_loss += criterion(y_, y.long()).item()
        pred = y_.argmax(axis=1)
        # pred = torch.round(y_)  # Apply sigmoid and round to get class 0 or 1

        y_true.extend(y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    return test_loss, accuracy, precision, recall, f1

class BatchDataset(Dataset):
    def __init__(self, data, tokenizer, targets=None, max_length=128):
        """
        初始化数据集
        :param data: 数据列表 (如函数代码)
        :param tokenizer: BERT分词器
        :param targets: 标签列表 (可选)
        :param max_length: 分词后的最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        加载单个样本并进行编码
        """
        text = self.data[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.int64)
            return input_ids, attention_mask, label
        return input_ids, attention_mask

    
class BertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2): # microsoft/codebert-base; microsoft/graphcodebert-base
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(args.model_dir).to(DEVICE)
        # self.tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes).to(DEVICE)

        self.count_parameters()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))
        pooled_output = outputs[1]
        x = self.dropout(pooled_output.to(DEVICE))
        logits = self.fc(x)
        # sig = torch.sigmoid(torch.flatten(logits))
        return logits
    
    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")

def process(args):
    # 数据集划分
    print(f"start loading metadata {args.dataset}")
    m1, m2, m3 = load_data(args.dataset)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)

    train_dataset = BatchDataset(m1['func'].tolist(), tokenizer, m1.target.values)
    val_dataset = BatchDataset(m2['func'].tolist(), tokenizer, m2.target.values)
    test_dataset = BatchDataset(m3['func'].tolist(), tokenizer, m3.target.values)

    batch_size = args.batch_size
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print(f"start initing model")
    model = BertClassifier().to(DEVICE)
    # import pdb;pdb.set_trace()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # def criterion(pred, label):
    #     return F.binary_cross_entropy(pred, label) 
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    print(f"start training")
    model_folder = os.path.join(args.output_path, f"BERT_{args.dataset}")
    os.makedirs(model_folder, exist_ok=True)

    run_name = f"algo:bert_data:{args.dataset}"
    wandb.init(project="vul-detect-f1", name=run_name, config={})
    wandb.config.update({
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "dataset": args.dataset,
        "device": args.device,
    })

    best_f1 = 0
    best_model_path = None

    for epoch in tqdm(range(1, args.num_epochs + 1), total=args.num_epochs, desc="Epoch"):
        # Training
        train_loss = train(model, DEVICE, train_dataset, criterion, optimizer, epoch)

        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, DEVICE, val_dataset, criterion)

        # Checkpoint saving if accuracy improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = f"{model_folder}/model_ep_new_{epoch}.tar"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, best_model_path)

        # Logging metrics for the current epoch
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "val/f1": val_f1
        })

    # Load the best model and evaluate on the test dataset
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, DEVICE, test_dataset, criterion)

        # Log final test metrics
        wandb.log({
            "epoch": epoch+1,
            "val/loss": test_loss,
            "val/acc": test_acc,
            "val/precision": test_precision,
            "val/recall": test_recall,
            "val/f1": test_f1
        })
    # for epoch in tqdm(range(1, args.num_epochs + 1), total=args.num_epochs, desc="Epoch"): 
    #     train_loss = train(model, DEVICE, train_dataset, criterion, optimizer, epoch)
    #     test_loss, acc, precision, recall, f1 = evaluate(model, DEVICE, val_dataset, criterion)
    #     if best_acc < acc:
    #             best_acc = acc
    #             model_name = f"{model_folder}/model_ep_new_{epoch}.tar"
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': test_loss}, model_name)
                
    #             # print("acc is: {:.4f}, best acc is {:.4f}".format(acc, best_acc))
    #             wandb.log({
    #                     "epoch": epoch,
    #                     "train/loss": train_loss,
    #                     "val/acc": acc,
    #                     "val/loss": test_loss,
    #                     "val/precision": precision,
    #                     "val/recall": recall,
    #                     "val/f1": f1
    #                 })
    #     # print(f"start testing")
    #     checkpoint = torch.load(model_name, map_location=DEVICE)
    #     state_dict = checkpoint['model_state_dict']
    #     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    #     model.load_state_dict(state_dict)
    #     test_loss, acc, precision, recall, f1 = evaluate(model, DEVICE, test_dataset, criterion)
    # wandb.log({
    #     "epoch": epoch,
    #     "train/loss": train_loss,
    #     "val/acc": acc,
    #     "val/loss": test_loss,
    #     "val/precision": precision,
    #     "val/recall": recall,
    #     "val/f1": f1
    # })

    wandb.finish()



if __name__ == "__main__":
    BERT_PATH = "/home/aikedaer/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
    CODEBERT_PATH = "/home/aikedaer/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39" 
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a CodeBERT-based classifier.")
    parser.add_argument("--dataset", type=str, default="devign",
                        help="Dataset to use for training and testing.")
    parser.add_argument("--model_dir", type=str, default=f"{BERT_PATH}",
                        help="Path to the pre-trained BERT model directory.")
    parser.add_argument("--device", type=str, default="cuda:7" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training and inference.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading.")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--output_path", type=str, default="bert/models/finetune", help="Path to save the best model.")

    args = parser.parse_args()
    
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set device
    DEVICE = torch.device(args.device)

    process(args)



# python bert.py --dataset cvefixes --device cuda:1
# python bert.py --dataset mvd  --device cuda:2
# python bert.py --dataset diversevul --device cuda:3
# python bert.py --dataset crossvul --device cuda:4
# python bert.py --dataset reveal 

