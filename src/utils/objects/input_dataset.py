import threading
from torch.utils.data import Dataset
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
import torch

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


class InputDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index].input
        return data

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=True)

from transformers import BertTokenizer
class StreamDataset0(Dataset):
    def __init__(self, dataset, CONFIG):
        self.config = CONFIG
        self.emb_size = CONFIG.vulberta.model.emb_size
        self.model_dir = CONFIG.vulberta.model.model_dir
        self.device = CONFIG.device
        
        self.vocab_path = CONFIG.vulberta.vocab_path #"vulberta/tokenizer/drapgh-vocab.json"
        self.merges_path = CONFIG.vulberta.merges_path # "vulberta/tokenizer/drapgh-merges.txt"
        # self.tokenizer = VulBertaTokenizer(self.vocab_path, self.merges_path, max_length=512).get_tokenizer()
        self.tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", truncation_side='right', model_max_length=512)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        # self.bert_model = RobertaModel.from_pretrained(self.model_dir).to(self.device)
        # self.embed_model = (self.tokenizer, self.bert_model, process.model.encode_input)

        self.dataset = dataset
        
        # self.dataset["nodes"] = self.dataset.progress_apply(lambda row: cpg.parse_to_nodes(row.cpg, CONFIG.embed.nodes_dim), axis=1)
        # self.dataset = self.dataset[self.dataset['nodes'].apply(len) > 0]
        # self.dataset = self.dataset.progress_apply(lambda row: prepare.nodes_to_input(row.nodes, row.target, row.func, self.config.embed.nodes_dim,
        #                                                                     self.embed_model, self.config.embed.edge_type, self.config), axis=1)

        # func, target, cpg, nodes => input

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index].input
        text = self.dataset.iloc[index].func
        # encoded = self.tokenizer.encode(text)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        data.input_ids = torch.tensor(encoded.input_ids.squeeze(0), dtype=torch.int64)
        data.attention_mask = torch.tensor(encoded.attention_mask.squeeze(0), dtype=torch.int64)
        # import pdb;pdb.set_trace()

        return data

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True)


class StreamDataset(Dataset):
    def __init__(self, dataset, CONFIG):
        self.config = CONFIG
        self.emb_size = CONFIG.vulberta.model.emb_size
        self.model_dir = CONFIG.vulberta.model.model_dir
        self.device = CONFIG.device
        
        self.vocab_path = CONFIG.vulberta.vocab_path #"vulberta/tokenizer/drapgh-vocab.json"
        self.merges_path = CONFIG.vulberta.merges_path # "vulberta/tokenizer/drapgh-merges.txt"
        self.tokenizer = VulBertaTokenizer(self.vocab_path, self.merges_path).get_tokenizer()
        
        # self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        # self.bert_model = RobertaModel.from_pretrained(self.model_dir).to(self.device)
        # self.embed_model = (self.tokenizer, self.bert_model, process.model.encode_input)

        self.dataset = dataset
        # self.dataset["nodes"] = self.dataset.progress_apply(lambda row: cpg.parse_to_nodes(row.cpg, CONFIG.embed.nodes_dim), axis=1)
        # self.dataset = self.dataset[self.dataset['nodes'].apply(len) > 0]
        # self.dataset = self.dataset.progress_apply(lambda row: prepare.nodes_to_input(row.nodes, row.target, row.func, self.config.embed.nodes_dim,
        #                                                                     self.embed_model, self.config.embed.edge_type, self.config), axis=1)

        # func, target, cpg, nodes => input

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()
        data = self.dataset.iloc[index].input
        text = self.dataset.iloc[index].func
        encoded = self.tokenizer.encode(text)
        data.input_ids = torch.tensor(encoded.ids, dtype=torch.int64)
        # data.attention_mask = torch.tensor(encoded.mask, dtype=torch.int64)
        # data.text = text
        return data

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=True)

