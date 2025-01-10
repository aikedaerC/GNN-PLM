import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import RobertaModel
import re
import os

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import threading
from typing import List

# for vulberta
from clang import *
from tokenizers import NormalizedString,PreTokenizedString
from typing import List 
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import StripAccents,Replace
from tokenizers import processors
from tokenizers.processors import TemplateProcessing
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    
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
        encoded = self.tokenizer.encode(text)

        input_ids = torch.tensor(encoded.ids, dtype=torch.int64)

        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.int64)
            return input_ids, label
        return input_ids


class MyTokenizerOld:
    cidx = cindex.Index.create()
        
    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        ## Tokkenize using clang
        tok = []
        tu = self.cidx.parse('tmp.c',
                       args=[''],  
                       unsaved_files=[('tmp.c', str(normalized_string.original))],  
                       options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()
            
            if spelling == '':
                continue
            #spelling = spelling.replace(' ', '')
            tok.append(NormalizedString(spelling))

        return(tok)
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)

class MyTokenizer:
    cidx = cindex.Index.create()

    def __init__(self, timeout=5):  # 设置超时时间
        self.timeout = timeout

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """
        使用 Clang 对代码进行分词，增加超时机制
        """
        result = []
        exception = None

        def parse():
            nonlocal result, exception
            try:
                tok = []
                tu = self.cidx.parse(
                    'tmp.c',
                    args=[''],  
                    unsaved_files=[('tmp.c', str(normalized_string.original))],  
                    options=0
                )
                for t in tu.get_tokens(extent=tu.cursor.extent):
                    spelling = t.spelling.strip()
                    if spelling == '':
                        continue
                    tok.append(NormalizedString(spelling))
                result = tok
            except Exception as e:
                exception = e

        # 创建线程
        thread = threading.Thread(target=parse)
        thread.start()
        thread.join(self.timeout)  # 等待超时时间

        if thread.is_alive():  # 超时检查
            print(f"Timeout occurred while parsing: {normalized_string.original[:100]}...")
            thread.join(0)  # 跳过此任务
            return []
        if exception:
            print(f"Error during Clang parsing: {exception}")
            return []

        return result

    def pre_tokenize(self, pretok: PreTokenizedString):
        """
        对预分词字符串进行处理，调用 Clang 分词器
        """
        def preprocess_and_split(i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
            return self.clang_split(i, normalized_string)
        
        pretok.split(preprocess_and_split)

class VulBertaTokenizer:
    def __init__(self, vocab_path, merges_path, max_length=1024, pad_token="<pad>"):
        # Load pre-trained tokenizers
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.max_length = max_length
        self.pad_token = pad_token
        self.tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        vocab, merges = BPE.read_file(vocab=self.vocab_path, merges=self.merges_path)
        tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

        tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
        tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", 0),
                ("<pad>", 1),
                ("</s>", 2),
                ("<unk>", 3),
                ("<mask>", 4)
            ]
        )
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(
            direction='right', 
            pad_id=1, 
            pad_type_id=0, 
            pad_token=self.pad_token,
            length=self.max_length, 
            pad_to_multiple_of=None
        )

        return tokenizer

    def get_tokenizer(self):
        return self.tokenizer

class CNNClassifier(nn.Module):
    def __init__(self, EMBED_DIM, model_dir):
        super(CNNClassifier,self).__init__()
        
        pretrained_weights = RobertaModel.from_pretrained(model_dir).embeddings.word_embeddings.weight

        self.embed = nn.Embedding.from_pretrained(pretrained_weights,
                                                  freeze=True,
                                                  padding_idx=1)

        self.conv1 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(200*3,256) #500
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,2)

        self.count_parameters()
    
    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0,2,1)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        x1 = F.max_pool1d(x1, x1.shape[2])
        x2 = F.max_pool1d(x2, x2.shape[2])
        x3 = F.max_pool1d(x3, x3.shape[2])
        
        x = torch.cat([x1,x2,x3],dim=1)
        x = x.flatten(1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # sig = torch.sigmoid(torch.flatten(logits))
        return logits
    
    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")

def select(result):
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
    for batch_idx, (x, y) in enumerate(train_loader): #, total=len(train_loader), desc=f"Training"):
        # if batch_idx == 196: # batch_size = 128
        #     import pdb;pdb.set_trace()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
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

    for batch_idx, (x, y)  in enumerate(test_loader): #, total=len(test_loader), desc=f"evaluating"):
        # if batch_idx == 77: # batch_size = 128
        #     import pdb;pdb.set_trace()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x)
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


def process(args):

    # 数据集划分
    print(f"start loading metadata {args.dataset}")
    m1, m2, m3 = load_data(args.dataset)

    vocab_path = "vulberta/tokenizer/drapgh-vocab.json"
    merges_path = "vulberta/tokenizer/drapgh-merges.txt"
    tokenizer = VulBertaTokenizer(vocab_path, merges_path).get_tokenizer()

    train_dataset = BatchDataset(m1['func'].tolist(), tokenizer, m1.target.values)
    val_dataset = BatchDataset(m2['func'].tolist(), tokenizer, m2.target.values)
    test_dataset = BatchDataset(m3['func'].tolist(), tokenizer, m3.target.values)

    batch_size = args.batch_size
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    print(f"start initing model")
    model = CNNClassifier(args.embed_dim, args.model_dir)
    #model.embed.weight.data[UNK_IDX] = torch.zeros(EMBED_DIM)
    model.embed.weight.data[args.pad_idx] = torch.zeros(args.embed_dim)
    model.to(DEVICE)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # def criterion(pred, label):
    #     return F.binary_cross_entropy(pred, label) 
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    

    def training(args):
        print(f"start training")
        run_name = f"algo:vulberta_data:{args.dataset}"
        wandb.init(project="vul-detect-f1", name=run_name, config={})
        wandb.config.update({
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "dataset": args.dataset,
            "device": args.device,
        })

        model_folder = os.path.join(args.output_path, f"VulBERTa-CNN_{args.dataset}")
        os.makedirs(model_folder, exist_ok=True)

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
                "epoch": epoch + 1,
                "val/loss": test_loss,
                "val/acc": test_acc,
                "val/precision": test_precision,
                "val/recall": test_recall,
                "val/f1": test_f1
            })
        wandb.finish()

    def testing(args):
        # print('Testing started....... old')

        if args.dataset == "devign":
            ckpt_file = f"/home/aikedaer/mydata/VulBERTa/models/finetune/VB-CNN_{args.dataset}/model_ep_17.tar"
        elif args.dataset == "mvd":
            ckpt_file = f"/home/aikedaer/mydata/VulBERTa/models/finetune/VB-CNN_{args.dataset}/model_ep_27.tar"
        elif args.dataset == "draper":
            ckpt_file = f"/home/aikedaer/mydata/VulBERTa/models/finetune/VB-CNN_{args.dataset}/model_ep_15.tar"
        elif args.dataset == "reveal":
            ckpt_file = f"/home/aikedaer/mydata/VulBERTa/models/finetune/VB-CNN_{args.dataset}/model_ep_4.tar"
        elif args.dataset == "vuldeepecker":
            ckpt_file = f"/home/aikedaer/mydata/VulBERTa/models/finetune/VB-CNN_{args.dataset}/model_ep_11.tar"
        elif args.dataset == "d2a":
            ckpt_file = f"/home/aikedaer/mydata/VulBERTa/models/finetune/VB-CNN_{args.dataset}/model_ep_4.tar"
        
        ## Testing
        checkpoint = torch.load(ckpt_file, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        # Remove 'module.' prefix if it's there (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        test_loss, acc, precision, recall, f1 = evaluate(model, DEVICE, test_dataset, criterion)
        print({
            "dataset": args.dataset,
            "algo": "vulberta",
            "val/acc": acc,
            "val/loss": test_loss,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1
        })

    if args.mode == "train":
        training(args)
    elif args.mode == "test":
        testing(args)


if __name__ == "__main__":

    BERT_PATH = "/home/aikedaer/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
    CODEBERT_PATH = "/home/aikedaer/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"    
    VulBERTa_path = "vulberta/models/pretrain/VulBERTa/"
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a CodeBERT-based classifier.")
    parser.add_argument("--dataset", type=str, default="devign",
                        help="Dataset to use for training and testing.")
    parser.add_argument("--model_dir", type=str, default=f"{VulBERTa_path}",
                        help="Path to the pre-trained BERT model directory.")
    parser.add_argument("--device", type=str, default="cuda:7" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training and inference.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading.")
    parser.add_argument("--embed_dim", type=int, default=768, help="embed dim.")
    parser.add_argument("--pad_idx", type=int, default=1, help="pad idx.")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--output_path", type=str, default="vulberta/models/finetune", help="Path to save the best model.")
    parser.add_argument("--mode", type=str, default="train", help="train or test")

    args = parser.parse_args()

    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import os
    os.environ["WANDB_MODE"] = "disabled"

    # Set device
    DEVICE = torch.device(args.device)

    process(args)

# ["crossvul", "cvefixes", "mvd", "diversevul"]
# python vulberta.py --dataset crossvul --device cuda:5
# python vulberta.py --dataset cvefixes --device cuda:0
# python vulberta.py --dataset mvd --device cuda:3
# python vulberta.py --dataset diversevul --device cuda:4
# python vulberta.py --dataset reveal # is enough 