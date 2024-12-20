import argparse
import os
import re
import warnings
import json
import time
import numpy as np
import pandas as pd
import torch
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from pprint import pprint

warnings.filterwarnings('ignore') 

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cleaner(code):
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

    return m1, m2, m3


def preprocess(data_file, vocab_file, padding_size, test=False):
    x_text, y = data_file["func"].tolist(), data_file["target"].tolist()
    if not test:
        text_preprocessor = preprocessing.text.Tokenizer(oov_token="<UNK>")
        text_preprocessor.fit_on_texts(x_text)
        x = text_preprocessor.texts_to_sequences(x_text)
        word_dict = text_preprocessor.word_index
        json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
        vocab_size = len(word_dict) + 1
        x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size, padding='post', truncating='post')
        return x, y, vocab_size
    else:
        word_dict = json.load(open(vocab_file, 'r'))
        x = [[word_dict.get(word, 1) for word in sentence.split()] for sentence in x_text]
        x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size, padding='post', truncating='post')
        return x, y


def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters, filter_sizes, regularizers_lambda, dropout_rate):
    inputs = keras.Input(shape=(feature_size,), name='input_data')
    embed = keras.layers.Embedding(vocab_size, embed_size, input_length=feature_size, name='embedding')(inputs)
    embed = keras.layers.Reshape((feature_size, embed_size, 1), name='add_channel')(embed)

    pool_outputs = []
    for filter_size in map(int, filter_sizes.split(',')):
        conv = keras.layers.Conv2D(num_filters, (filter_size, embed_size), activation='relu', padding='valid')(embed)
        pool = keras.layers.MaxPool2D(pool_size=(feature_size - filter_size + 1, 1), padding='valid')(conv)
        pool_outputs.append(pool)

    pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1)
    pool_outputs = keras.layers.Flatten()(pool_outputs)
    pool_outputs = keras.layers.Dropout(dropout_rate)(pool_outputs)

    outputs = keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(regularizers_lambda))(pool_outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train(x_train, y_train, vocab_size, feature_size, args, save_path):
    model = TextCNN(vocab_size, feature_size, args.embed_size, args.num_classes, args.num_filters, args.filter_sizes, 0.01, args.dropout_rate)
    model.compile(tf.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(),
                        metrics=[['accuracy'],
                        keras.metrics.Precision(),
                        keras.metrics.Recall(),
                        keras.metrics.F1Score()])
    model.summary()
    # import pdb;pdb.set_trace()qw
    y_train = tf.one_hot(y_train, args.num_classes)
    history = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_split=args.fraction_validation, shuffle=True)
    model.save(save_path)
    pprint(history.history)


def test(model, x_test, y_true, num_classes):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    print('Test set: Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
    accuracy * 100, precision * 100, recall * 100, f1 * 100))



def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and test a TextCNN model.")
    parser.add_argument("--dataset", type=str, default="devign", help="Dataset name (e.g., draper, devign, etc.)")
    parser.add_argument("--padding_size", type=int, default=128, help="Padding size for text sequences.")
    parser.add_argument("--embed_size", type=int, default=512, help="Embedding size for the model.")
    parser.add_argument("--num_filters", type=int, default=128, help="Number of filters for convolutional layers.")
    parser.add_argument("--filter_sizes", type=str, default="3,4,5", help="Comma-separated filter sizes.")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the model.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--fraction_validation", type=float, default=0.05, help="Fraction of validation data.")
    parser.add_argument("--results_dir", type=str, default="textcnn/", help="Directory to save results.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    result_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    # import pdb;pdb.set_trace()
    m1, m2, m3 = load_data(args.dataset)
    args.fraction_validation = len(m2)/(len(m1)+len(m2))
    m1 = pd.concat([m1, m2], ignore_index=True)
    x_train, y_train, vocab_size = preprocess(m1, os.path.join(result_dir, "vocab.json"), args.padding_size)
    model_path = os.path.join(result_dir, f"TextCNN_{args.dataset}.h5")

    train(x_train, y_train, vocab_size, args.padding_size, args, model_path)

    x_test, y_test = preprocess(m3, os.path.join(result_dir, "vocab.json"), args.padding_size, test=True)
    model = load_model(model_path)
    test(model, x_test, y_test, args.num_classes)


if __name__ == "__main__":
    main()

# "crossvul", "cvefixes", "mvd", "diversevul"
# export CUDA_VISIBLE_DEVICES=3;python textcnn.py --dataset crossvul  
# export CUDA_VISIBLE_DEVICES=3;python textcnn.py --dataset cvefixes
# export CUDA_VISIBLE_DEVICES=3;python textcnn.py --dataset mvd
# export CUDA_VISIBLE_DEVICES=3;python textcnn.py --dataset diversevul
# python textcnn.py --dataset reveal