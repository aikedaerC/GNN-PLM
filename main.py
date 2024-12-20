# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""

import argparse
import copy
import gc
import shutil
from argparse import ArgumentParser
import re 
from gensim.models.word2vec import Word2Vec
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from configs import CONFIG
import src.data as data
import src.prepare as prepare
import src.process as process
from src.utils.objects.input_dataset import InputDataset, StreamDataset
import src.utils.functions.cpg as cpg
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from src.sam.bypass_bn import disable_running_stats, enable_running_stats
from src.sam import SAM
import wandb
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


PATHS = CONFIG.paths
FILES = CONFIG.files

PATHS.cpg = os.path.join(PATHS.cpg, CONFIG.create.dataset)
PATHS.tokens = os.path.join(PATHS.tokens, CONFIG.create.dataset)
PATHS.input = os.path.join(PATHS.input, CONFIG.create.dataset)
PATHS.joern = os.path.join(PATHS.joern, CONFIG.create.dataset)
try: 
    PATHS.w2v   = os.path.join(PATHS.w2v, CONFIG.create.dataset)
    os.makedirs(PATHS.w2v, exist_ok=True)
except:
    pass

os.makedirs(PATHS.cpg, exist_ok=True)
os.makedirs(PATHS.tokens, exist_ok=True)
os.makedirs(PATHS.input, exist_ok=True)


def select(dataset, filter_column_value):
    result = dataset # dataset.loc[dataset['project'] == filter_column_value.project]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    # result = result.head(200)
    return result

def cleaner(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    code = re.sub(r'\n|\t', '', code)
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
    # raw = pd.concat([m1,m2,m3], ignore_index=True)
    return m1,m2,m3


def create_task():
    """
    data.pkl -> temp.c -> cpg.bin -> cpg.json -> slices_input.pkl
    """
    context = CONFIG.create
    m1,m2,m3 = load_data(context.dataset)
    for mode, raw in zip(["train", "valid", "test"],[m1,m2,m3]):
        cpg_path = os.path.join(PATHS.cpg, mode)
        os.makedirs(cpg_path, exist_ok=True)

        filtered = data.apply_filter(raw, lambda dx: select(dx, context.filter_column_value))
        filtered = data.clean(filtered)

        slices = data.slice_frame(filtered, context.slice_size)
        slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]
        
        cpg_files = []
        # Create CPG binary files
        for s, slice in slices:
            cpg_file = os.path.join(cpg_path,f"{s}_{FILES.cpg}.bin")
            if not os.path.exists(cpg_file):
                data.to_files(slice, PATHS.joern)
                cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, cpg_path, f"{s}_{FILES.cpg}")
                print(f"Dataset {s} to cpg.")
                if os.path.exists(PATHS.joern):
                    shutil.rmtree(PATHS.joern)
            cpg_files.append(f"{s}_{FILES.cpg}.bin")
        # Create CPG with graphs json files
        # import pdb;pdb.set_trace()
        json_files = prepare.joern_create(context.joern_cli_dir, cpg_path, cpg_path, cpg_files)
        # import pdb;pdb.set_trace()
        for (s, slice), json_file in zip(slices, json_files):
            graphs = prepare.json_process(cpg_path, json_file)
            if graphs is None:
                print(f"Dataset chunk {s} not processed.")
                continue
            dataset = data.create_with_index(graphs, ["Index", "cpg"])
            dataset = data.inner_join_by_index(slice, dataset)

            print(f"Writing cpg dataset chunk {s}.")
            data.write(dataset, cpg_path, f"{s}_{FILES.cpg}.pkl")
            del dataset
            gc.collect()


def embed_task():
    context = CONFIG.embed
    # Tokenize source code into tokens
    for mode in ["train", "valid", "test"]:
        cpg_path = os.path.join(PATHS.cpg, mode)
        cpg_input = os.path.join(PATHS.input, mode)
        os.makedirs(cpg_input, exist_ok=True)
        dataset_files = data.get_directory_files(cpg_path)
        # ########################################################################################## #
        if context.embed_type in ["w2v", "vulberta", "vulberta_sam"]:
            w2vmodel = Word2Vec(**context.word2vec_args)                                             #
            w2v_init = True                                                                           #
        # ########################################################################################## #
        elif context.embed_type in ["bert", "sam_bert"]:
            tokenizer_bert = RobertaTokenizer.from_pretrained(CONFIG.bertggcn.model.model_dir)
            bert_model = RobertaModel.from_pretrained(CONFIG.bertggcn.model.model_dir).to(CONFIG.device)
            embed_model = (tokenizer_bert, bert_model, process.model.encode_input, CONFIG.bertggcn.model)

        for pkl_file in tqdm(dataset_files, desc="Processing cpg pkl", unit="file"):
            file_name = pkl_file.split(".")[0]
            # import pdb;pdb.set_trace()
            cpg_dataset = data.load(cpg_path, pkl_file)
            tokens_dataset = data.tokenize(cpg_dataset) 
            # data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
            # ########################################################################################## #
            # word2vec used to learn the initial embedding of each token     
            if context.embed_type in ["w2v", "vulberta", "vulberta_sam"]:                                                              #
                w2vmodel.build_vocab(sentences=tokens_dataset.tokens, update=not w2v_init)               #
                w2vmodel.train(tokens_dataset.tokens, total_examples=w2vmodel.corpus_count, epochs=1)    #
                if w2v_init:                                                                             #
                    w2v_init = False                                                                     #
            # ########################################################################################## #
            # Embed cpg to node representation and pass to graph data structure
            if context.embed_type in ["w2v", "vulberta", "vulberta_sam"]:
                embed_model = w2vmodel.wv

            tqdm.pandas()
            cpg_dataset["nodes"] = cpg_dataset.progress_apply(lambda row: cpg.parse_to_nodes(row.cpg, context.nodes_dim), axis=1)
            # remove rows with no nodes
            cpg_dataset = cpg_dataset[cpg_dataset['nodes'].apply(len) > 0]
            cpg_dataset["input"] = cpg_dataset.progress_apply(lambda row: prepare.nodes_to_input(row.nodes, row.target, context.nodes_dim,
                                                                                        embed_model, context.edge_type, CONFIG), axis=1)
            
            data.drop(cpg_dataset, ["nodes"])
            print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")

            data.write(cpg_dataset[["input", "func"]], cpg_input, f"{file_name}_{FILES.input}")
            del cpg_dataset
            gc.collect()
        # ########################################################################################## #
        if context.embed_type in ["w2v", "sam_w2v"]:
            print("Saving w2vmodel.")                                                                #
            w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")                                                #
        # ########################################################################################## #

def process_task(args):
    MODEL_DIR = "/home/aikedaer/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"
    context = CONFIG.process
    model_path = os.path.join(PATHS.model,CONFIG.create.dataset)
    os.makedirs(model_path, exist_ok=True)
    model_path = model_path + "/" + FILES.model

    if CONFIG.embed.embed_type == "w2v":
        devign = CONFIG.devign
        model = process.Devign(devign.model.gated_graph_conv_args, devign.model.conv_args, devign.model.emb_size, CONFIG.device)
        optimizer = optim.Adam(model.parameters(), lr=devign.learning_rate, weight_decay=devign.weight_decay)
        def criterion(pred, label):
            return F.binary_cross_entropy(pred, label) #  + F.l1_loss(pred, label) * devign.loss_lambda

        train_dataset = data.loads(os.path.join(PATHS.input, "train"))
        valid_dataset = data.loads(os.path.join(PATHS.input, "valid"))
        test_dataset = data.loads(os.path.join(PATHS.input, "test"))

        train_loader = InputDataset(train_dataset).get_loader(context.batch_size, shuffle=context.shuffle)
        val_loader = InputDataset(valid_dataset).get_loader(context.batch_size, shuffle=context.shuffle)
        test_loader = InputDataset(test_dataset).get_loader(context.batch_size, shuffle=context.shuffle)


    elif CONFIG.embed.embed_type == "bert":
        bertggcn = CONFIG.bertggcn
        model = process.BertGGCN(bertggcn.model.gated_graph_conv_args, bertggcn.model.conv_args, bertggcn.model.emb_size, MODEL_DIR, CONFIG.device)
        optimizer = optim.Adam(model.parameters(), lr=bertggcn.learning_rate, weight_decay=bertggcn.weight_decay)
        def criterion(pred, label):
            return F.binary_cross_entropy(pred, label) #  + F.l1_loss(pred, label) * bertggcn.loss_lambda
            
        train_dataset = data.loads(os.path.join(PATHS.input, "train"))
        valid_dataset = data.loads(os.path.join(PATHS.input, "valid"))
        test_dataset = data.loads(os.path.join(PATHS.input, "test"))

        train_loader = InputDataset(train_dataset).get_loader(context.batch_size, shuffle=context.shuffle)
        val_loader = InputDataset(valid_dataset).get_loader(context.batch_size, shuffle=context.shuffle)
        test_loader = InputDataset(test_dataset).get_loader(context.batch_size, shuffle=context.shuffle)


    elif CONFIG.embed.embed_type == "vulberta":
        model = process.VulBertaGGCN(CONFIG)
        model.embed.weight.data[CONFIG.vulberta.model.pad_idx] = torch.zeros(CONFIG.vulberta.model.emb_size)
        model.to(CONFIG.device)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG.vulberta.learning_rate, weight_decay=CONFIG.vulberta.weight_decay)
        criterion = nn.CrossEntropyLoss().to(CONFIG.device)
        loader_dict = {}
        for mode in ["train", "valid", "test"]:
            dataset_ = data.loads(os.path.join(PATHS.input, mode))
            loader_dict[mode] = StreamDataset(dataset_, CONFIG).get_loader(CONFIG.process.batch_size, shuffle=CONFIG.process.shuffle)

        train_loader = loader_dict["train"]
        val_loader = loader_dict["valid"]
        test_loader = loader_dict["test"]

    
    model_test = copy.deepcopy(model)

    if args.mode == "train":
        run_name = f"algo:devign-{CONFIG.embed.embed_type}_data:{CONFIG.create.dataset}"
        wandb.init(project="vul-detect", name=run_name, config={})

        best_f1 = 0.0
        for epoch in tqdm(range(1, context.epochs + 1), desc="Epochs:"):
            train_loss = train(model, CONFIG.device, train_loader, optimizer, criterion, epoch)
            test_loss, accuracy, precision, recall, f1 = evaluate(model, CONFIG.device, val_loader, criterion)
            if best_f1 < f1:
                best_f1 = f1
                torch.save(model.state_dict(), args.path)
            # print("acc is: {:.4f}, best acc is {:.4f}".format(accuracy, best_f1))
            wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/acc": accuracy,
                    "val/loss": test_loss,
                    "val/precision": precision,
                    "val/recall": recall,
                    "val/f1": f1
                })
    model_test.load_state_dict(torch.load(args.path))
    test_loss, accuracy, precision, recall, f1 = evaluate(model_test, CONFIG.device, test_loader, criterion)
    wandb.log({
        "epoch": context.epochs + 1,
        "train/loss": train_loss,
        "val/acc": accuracy,
        "val/loss": test_loss,
        "val/precision": precision,
        "val/recall": recall,
        "val/f1": f1
    })

    wandb.finish()
   
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="training"):
        batch.to(device)
        y_pred = model(batch)
        model.zero_grad()
        loss = criterion(y_pred, batch.y.long()) 
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

    for batch_idx, batch in enumerate(test_loader): #, total=len(test_loader), desc=f"evaluating"):
        batch.to(device)
        with torch.no_grad():
            y_ = model(batch)
        test_loss += criterion(y_, batch.y.long()).item()
        
        if len(y_.shape)==2:
            pred = y_.argmax(axis=1)
        elif len(y_.shape)==1:
            pred = torch.round(y_)  # Apply sigmoid and round to get class 0 or 1

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    return test_loss, accuracy, precision, recall, f1

def main():
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-cpg', '--cpg', action='store_true', help='Specify to perform CPG generation task')
    parser.add_argument('-embed', '--embed', action='store_true', help='Specify to perform Embedding generation task')
    parser.add_argument('-mode', '--mode', default="train", help='Specify the mode (e.g., train, test)')
    parser.add_argument('-path', '--path', default="ckpts/default.pth", help='Specify the path for the model')
    parser.add_argument('-p', '--process', action='store_true')

    args = parser.parse_args()

    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.cpg:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(args)

if __name__ == "__main__":
    main()

# python main.py -cpg -embed -p -mode train -path ckpts/w2v_crossvul.pth
# python main.py -p -mode train -path ckpts/w2v_mvd.pth
# python main.py -p -mode train -path ckpts/w2v_draper.pth
# python main.py -p -mode train -path ckpts/w2v_vuldeepecker.pth
# python main.py -p -mode train -path ckpts/w2v_reveal.pth


# python main.py -p -mode train -path ckpts/bert_devign.pth
# python main.py -p -mode train -path ckpts/bert_mvd.pth
# python main.py -p -mode train -path ckpts/bert_draper.pth
# python main.py -p -mode train -path ckpts/bert_vuldeepecker.pth
# python main.py -p -mode train -path ckpts/bert_reveal.pth