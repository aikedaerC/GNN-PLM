from box import Box

config_data={
    "create" : {
        "filter_column_value": {"project": "qemu"},
        "slice_size": 200,
        "joern_cli_dir": "joern/joern-cli/",
        "dataset": "devign"
    },

    "paths" : {
        "cpg": "data/cpg/",
        "joern": "data/joern/",
        "raw": "/home/aikedaer/mydata/data/vuldata/Devign/",
        "input": "data/input/",
        "model": "data/model/input/",
        "tokens": "data/tokens/",
        "w2v": "data/w2v/"
    },

    "files" : {
        "raw": "function.json",
        "cpg": "cpg",
        "tokens": "tokens.pkl",
        "w2v": "w2v.model",
        "input": "input.pkl",
        "model": "checkpoint.pt"
    },

    "embed" : {
        "embed_type": "bert", 
        "nodes_dim": 205,
        "word2vec_args": { 
            "size": 100,
            "alpha": 0.01,
            "window": 5,
            "min_count": 3,
            "sample": 1e-5,
            "workers": 4,
            "sg": 1,
            "hs": 0,
            "negative": 5
        },
        "edge_type": "Ast" #Cpg, Ast
    },

    "process" : {
        "epochs": 50,
        "patience": 10,
        "batch_size": 128,
        "dataset_ratio": 0.2,
        "shuffle": False
    },
    "devign" : {
        "learning_rate": 1e-4,
        "weight_decay": 1.3e-6,
        "loss_lambda": 1.3e-6,
        "model": {
            "gated_graph_conv_args": {"out_channels": 200, "num_layers": 6, "aggr": "add", "bias": True},
            "conv_args": {
                "conv1d_1": {"in_channels": 205, "out_channels": 50, "kernel_size": 3, "padding": 1},
                "conv1d_2": {"in_channels": 50, "out_channels": 20, "kernel_size": 1, "padding": 1},
                "maxpool1d_1": {"kernel_size": 3, "stride": 2},
                "maxpool1d_2": {"kernel_size": 2, "stride": 2}
            },
            "emb_size": 101
        }
    },
    "bertggcn" : {
        "learning_rate": 1e-3,
        "weight_decay": 1.3e-6,
        "loss_lambda": 1.3e-6,
        "model": {
            "model_dir": "/home/aikedaer/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39",
            "gated_graph_conv_args": {"out_channels": 200, "num_layers": 6, "aggr": "add", "bias": True},
            "conv_args": {
                "conv1d_1": {"in_channels": 205, "out_channels": 50, "kernel_size": 3, "padding": 1},
                "conv1d_2": {"in_channels": 50, "out_channels": 20, "kernel_size": 1, "padding": 1},
                "maxpool1d_1": {"kernel_size": 3, "stride": 2},
                "maxpool1d_2": {"kernel_size": 2, "stride": 2}
            },
            "emb_size": 101
        }
    },
    "vulberta": {
        "learning_rate": 2e-5,
        "weight_decay": 1.3e-6,
        "loss_lambda": 1.3e-6,
        "model": {
            "pad_idx": 1,
            "model_dir": "/home/aikedaer/mydata/devign/baselines/vulberta/models/pretrain/VulBERTa/",
            "gated_graph_conv_args": {"out_channels": 200, "num_layers": 6, "aggr": "add", "bias": True},
            "emb_size": 768
        },
        "vocab_path": "/home/aikedaer/mydata/devign/baselines/vulberta/tokenizer/drapgh-vocab.json",
        "merges_path": "/home/aikedaer/mydata/devign/baselines/vulberta/tokenizer/drapgh-merges.txt"
    },
    "device" : "cuda:1" # 6.5.4.3.1
}

# EMBED_TYPE="w2v"
# EMBED_TYPE="bert"
# EMBED_TYPE="vulberta"
EMBED_TYPE="vulberta_sam"

# ["crossvul", "cvefixes", "mvd", "diversevul", "reveal"]

config_data["create"]["dataset"] = "mvd" 

if EMBED_TYPE == "w2v":
    config_data["embed"]["embed_type"] = "w2v"
    config_data["learning_rate"] = config_data["devign"]["learning_rate"]
    config_data["paths"] = {
        "cpg": "data/cpg/",
        "joern": "data/joern/",
        "raw": "/home/aikedaer/mydata/data/vuldata/Devign/",
        "input": "data/input/w2v/",
        "model": "data/model/w2v/",
        "tokens": "data/tokens/w2v/",
        "w2v": "data/w2v/"
    }
elif EMBED_TYPE == "bert": 
    config_data["embed"]["embed_type"] = "bert" 
    config_data["learning_rate"] = config_data["bertggcn"]["learning_rate"]
    config_data["paths"] = {
        "cpg": "data/cpg/",
        "joern": "data/joern/",
        "raw": "/home/aikedaer/mydata/data/vuldata/Devign/",
        "input": "data/input/bert/",
        "model": "data/model/bert/",
        "tokens": "data/tokens/bert/",
    }
elif EMBED_TYPE == "vulberta": 
    config_data["embed"]["embed_type"] = "vulberta"
    config_data["learning_rate"] = config_data["vulberta"]["learning_rate"]
    config_data["paths"] = {
        "cpg": "data/cpg/",
        "joern": "data/joern/",
        "raw": "/home/aikedaer/mydata/data/vuldata/Devign/",
        "input": "data/input/w2v/",
        "model": "data/model/w2v/",
        "tokens": "data/tokens/w2v/",
    }
elif EMBED_TYPE == "vulberta_sam": 
    config_data["embed"]["embed_type"] = "vulberta_sam"
    config_data["learning_rate"] = config_data["vulberta"]["learning_rate"]
    config_data["paths"] = {
        "cpg": "data/cpg/",
        "joern": "data/joern/",
        "raw": "/home/aikedaer/mydata/data/vuldata/Devign/",
        "input": "data/input/w2v/",
        "model": "data/model/w2v/",
        "tokens": "data/tokens/w2v/",
    }

CONFIG = Box(config_data)
