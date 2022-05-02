import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from transformers import AutoModel, AutoTokenizer
#display options
pd.set_option('display.max_colwidth', None)
"""
超参数
"""
hyperparameters = {
    "max_length": 416,
    "padding": "max_length",
    "return_offsets_mapping": True,
    "truncation": "only_second",
    "debug": False,

    "model_name": ("../input/layoutlm/BiomedNLP-PubMedBERT"
                   "-base-uncased-abstract-fulltext"),
    "dropout": 0.2,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-5,
    "weight_decay": 0.01,
    "betas": (0.9, 0.999),
    "lr": 1e-5,

    "seed": 1268,
    "test_batch_size": 8,
    "epochs": 6,

    "apex": True,
    "eps": 1e-6,

    "n_fold": 5,
    "trn_fold": [1, 2, 3, 4, 5]
}
if hyperparameters['debug']:
    hyperparameters['epochs'] = 2
    hyperparameters['trn_fold'] = [1, 2]

"""
预测帮助函数
"""
def get_location_predictions(preds, offset_mapping, sequence_ids, test=False):
    all_predictions = []
    for pred, offsets, seq_ids in zip(preds, offset_mapping, sequence_ids):
        if not test:
            pred = 1 / (1 + np.exp(-pred))
        else:
            pass
        start_idx = None
        end_idx = None
        current_preds = []
        for pred, offset, seq_id in zip(pred, offsets, seq_ids):
            if seq_id is None or seq_id == 0:
                continue

            if pred > 0.5:
                if start_idx is None:
                    start_idx = offset[0]
                end_idx = offset[1]
            elif start_idx is not None:
                # 增加if语句筛选0 0的状况
                if test:
                    if start_idx == 0 and end_idx == 0:
                        start_idx = None
                        continue
                    current_preds.append(f"{start_idx} {end_idx}")
                else:
                    current_preds.append((start_idx, end_idx))
                start_idx = None
        if test:
            all_predictions.append("; ".join(current_preds))
        else:
            all_predictions.append(current_preds)

    return all_predictions

"""
分词
"""
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(hyperparameters['model_name'])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_URL = "../input/nbme-score-clinical-patient-notes"
TRAIN_URL = "../input/pubmedbert"

"""
数据类
"""
def create_test_df():
    feats = pd.read_csv(f"{BASE_URL}/features.csv")
    notes = pd.read_csv(f"{BASE_URL}/patient_notes.csv")
    test = pd.read_csv(f"{BASE_URL}/test.csv")

    merged = test.merge(notes, how="left")
    merged = merged.merge(feats, how="left")

    def process_feature_text(text):
        return text.replace("-OR-", ";-").replace("-", " ").replace("I-year", "1-year")

    merged["feature_text"] = [process_feature_text(x) for x in merged["feature_text"]]
    return merged


class SubmissionDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.loc[idx]
        tokenized = self.tokenizer(
            example["feature_text"],
            example["pn_history"],
            truncation=self.config['truncation'],
            max_length=self.config['max_length'],
            padding=self.config['padding'],
            return_offsets_mapping=self.config['return_offsets_mapping']
        )
        tokenized["sequence_ids"] = tokenized.sequence_ids()

        input_ids = np.array(tokenized["input_ids"])
        attention_mask = np.array(tokenized["attention_mask"])
        token_type_ids = np.array(tokenized["token_type_ids"])
        offset_mapping = np.array(tokenized["offset_mapping"])
        sequence_ids = np.array(tokenized["sequence_ids"]).astype("float16")

        return input_ids, attention_mask, token_type_ids, offset_mapping, sequence_ids


test_df = create_test_df()

submission_data = SubmissionDataset(test_df, tokenizer, hyperparameters)
submission_dataloader = DataLoader(submission_data,
                                   batch_size=hyperparameters['test_batch_size'],
                                   pin_memory=True,
                                   shuffle=False)
"""
模型类
"""
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config['model_name'])
        self.dropout = nn.Dropout(p=config['dropout'])
        self.config = config
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        logits = F.relu(self.fc1(outputs[0]))
        logits = F.relu(self.fc2(self.dropout(logits)))
        logits = self.fc3(self.dropout(logits)).squeeze(-1)
        return logits


model = CustomModel(hyperparameters).to(DEVICE)


"""
分折预测
"""
import gc
# 准备分折预测
predictions = []
for fold in hyperparameters['trn_fold']:
    model = CustomModel(hyperparameters).to(DEVICE)
    model.load_state_dict(torch.load(f"{TRAIN_URL}/nbme_pubmed_bert_fold{fold}.pth"))
    model.eval()
    # 初始化指标容器
    preds = []
    offsets = []
    seq_ids = []
    #     logits_container = []
    for batch in tqdm(submission_dataloader):
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        token_type_ids = batch[2].to(DEVICE)
        offset_mapping = batch[3]
        sequence_ids = batch[4]
        logits = model(input_ids, attention_mask, token_type_ids)
        #         logits_container.append(logits)
        preds.append(logits.sigmoid().detach().cpu().numpy())
        offsets.append(offset_mapping.numpy())
        seq_ids.append(sequence_ids.numpy())

    preds = np.concatenate(preds, axis=0)
    predictions.append(preds)
    offsets = np.concatenate(offsets, axis=0)
    seq_ids = np.concatenate(seq_ids, axis=0)

    del model, preds;
    gc.collect()
    torch.cuda.empty_cache()
# 五折总输出的平均
predictions = np.mean(predictions, axis=0)

# 输出
location_preds = get_location_predictions(predictions, offsets, seq_ids, test=True)
test_df["location"] = location_preds
test_df[["id", "location"]].to_csv("submission.csv", index=False)