import sys
sys.path.append("../input/transformers/src")
import transformers

print(f"Transformers version: {transformers.__version__}")


import os  # noqa
import re
import gc
import ast
import sys  # noqa
import pickle  # noqa
import random  # noqa
import itertools
import warnings
import itertools  # noqa
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython import embed  # noqa

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#辅助函数
if 'KAGGLE_URL_BASE' in set(os.environ.keys()):
    OUTPUT_DIR = Path('../temp')
    OUTPUT_DIR.mkdir(exist_ok=True)
else:
    OUTPUT_DIR = Path('')

def get_tokenizer(dir_name):
    tokenizer = AutoTokenizer.from_pretrained(INPUT_DIR / dir_name / 'tokenizer', trim_offsets=False)
    # checkpointがない場合は、エラーになる  # TODO: どう修正するか検討する
    return tokenizer


def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def format_spans(indices):
    segs = []
    left = last = None
    for i in indices:
        if left is None:
            left = last = i
        elif last + 1 == i:
            last = i
        else:
            # New segment
            segs.append('%d %d' % (left, last + 1))
            left = last = i

    if last is not None:
        segs.append('%d %d' % (left, last + 1))

    return ';'.join(segs)


def get_results(char_probs, threshold=0.5):
    locs = []
    for prob in char_probs:
        pn_history = prob['pn_history']
        y_prob = prob['character_level_probs']

        list_ = []
        i_begin = i_last = None
        last_space = False
        for i, (x, p) in enumerate(zip(pn_history, y_prob)):
            if p >= threshold:
                if i_begin is None:  # 先頭
                    if x not in (' ', '\n', '\r', '\t'):  # 文字がスペースじゃない
                        i_begin = i_last = i
                        list_.append(i)
                    else:
                        pass
                else:  # 先頭じゃない
                    assert i_last + 1 == i  # i_last + 1 と　i　が同一じゃなかったらエラー
                    i_last = i
                    list_.append(i)
                    if x in (' ', '\n', '\r', '\t'):
                        last_space = True
                    else:
                        last_space = False
            else:
                i_begin = i_last = None  # Negative; reset span
                if last_space:
                    list_.pop(-1)
                    last_space = False
        locs.append(format_spans(list_))

    return locs


def get_results_with_threshold_list(char_probs, threshold_list):
    locs = []
    for prob, threshold in zip(char_probs, threshold_list):
        pn_history = prob['pn_history']
        y_prob = prob['character_level_probs']

        list_ = []
        i_begin = i_last = None
        last_space = False
        for i, (x, p) in enumerate(zip(pn_history, y_prob)):
            if p >= threshold:
                if i_begin is None:  # 先頭
                    if x not in (' ', '\n', '\r', '\t'):  # 文字がスペースじゃない
                        i_begin = i_last = i
                        list_.append(i)
                    else:
                        pass
                else:  # 先頭じゃない
                    assert i_last + 1 == i  # i_last + 1 と　i　が同一じゃなかったらエラー
                    i_last = i
                    list_.append(i)
                    if x in (' ', '\n', '\r', '\t'):
                        last_space = True
                    else:
                        last_space = False
            else:
                i_begin = i_last = None  # Negative; reset span
                if last_space:
                    list_.pop(-1)
                    last_space = False

        locs.append(format_spans(list_))

    return locs

# ====================================================
# Dataset和Model
# ====================================================
def create_label(tokenizer, pn_history, annotation_length, location_list, max_len):
    encoded = tokenizer(pn_history,
                        add_special_tokens=True,
                        max_length=max_len,
                        padding="max_length",
                        return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return label

def df_merge(df, mode, preprocess_feature):
    # マージ
    patient_notes = pd.read_csv(INPUT_DIR / 'nbme-score-clinical-patient-notes' / "patient_notes.csv")
    features = pd.read_csv(INPUT_DIR / 'nbme-score-clinical-patient-notes' / "features.csv")
    if preprocess_feature:
        features['feature_text'] = features['feature_text'].apply(process_feature_text)
    df = df.merge(features, how='left', on=["case_num", "feature_num"])
    df = df.merge(patient_notes, how="left", on=['case_num', 'pn_num'])

    if mode != 'test':
        # データ形式の変換
        df['anno_list'] = df['annotation'].apply(ast.literal_eval)
        df['loc_list'] = df['location'].apply(ast.literal_eval)
        df['anno_length'] = df['anno_list'].apply(len)

    return df

class CustomModelTest(nn.Module):
    def __init__(self, model_dir, config_path=None, pretrained=False, output_dim=1):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(model_dir, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_dir, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        # self.fc_dropout = nn.Dropout(args.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, output_dim)

    def forward(self, tokens, attention_mask, token_type_ids):
        outputs = self.model(
            tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        feature = outputs[0]  # last_hidden_states

        output = self.fc(feature)
        return output


def create_labels_for_scoring_fix(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    # 'loc_list'から作成するように変更
    df['location_for_create_labels'] = [ast.literal_eval('[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'loc_list']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[["{new_lst}"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


def prepare_input(tokenizer, pn_history, feature_text, return_pn_history=False,
                  return_offsets_mapping=False, padding='max_length', max_len=500):
    inputs = tokenizer(pn_history,
                       feature_text,
                       add_special_tokens=True,
                       max_length=max_len,
                       truncation=True,
                       padding=padding,
                       return_offsets_mapping=return_offsets_mapping)
    if return_pn_history:
        inputs['pn_history'] = pn_history
    return inputs
class TestDataset(Dataset):
    def __init__(self, tokenizer, df, max_len, return_label=False):
        self.tokenizer = tokenizer
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        if return_label:
            self.annotation_lengths = df['anno_length'].values
            self.locations = df['loc_list'].values
        self.return_label = return_label
        self.max_len = max_len
        if return_label:
            self.labels = []
            for i in range(len(df)):
                label = create_label(
                    tokenizer,
                    df.loc[i, 'pn_history'],
                    df.loc[i, 'anno_length'],
                    df.loc[i, 'loc_list'],
                    max_len
                    )
                self.labels.append(label)

    def __len__(self):
        return len(self.pn_historys)

    def __getitem__(self, idx):
        inputs = prepare_input(self.tokenizer,
                               self.pn_historys[idx],
                               self.feature_texts[idx],
                               return_pn_history=True,
                               return_offsets_mapping=True,
                               padding=False,
                               max_len=self.max_len
                               )

        if not self.return_label:
            return inputs
        else:
            inputs['label'] = self.labels[idx]
        return inputs


class Collate:
    def __init__(self, tokenizer, return_label=False):
        self.tokenizer = tokenizer
        self.return_label = return_label

    def __call__(self, batch, label=None):
        output = dict()
        output["offset_mapping"] = [sample["offset_mapping"] for sample in batch]
        output["pn_history"] = [sample["pn_history"] for sample in batch]

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["token_type_ids"] = [sample["token_type_ids"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(token_id) for token_id in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
            output["token_type_ids"] = [s + (batch_max - len(s)) * [0] for s in output["token_type_ids"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]
            output["token_type_ids"] = [(batch_max - len(s)) * [0] + s for s in output["token_type_ids"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        output["token_type_ids"] = torch.tensor(output["token_type_ids"], dtype=torch.long)

        if self.return_label:
            labels = torch.tensor([sample['label'] for sample in batch], dtype=torch.float)
            output['labels'] = labels[:, :batch_max]

        return output


def make_oof(device, model_dir, max_len, num_fold, ckpt_path, output_dim):

    model = CustomModelTest(model_dir, config_path=INPUT_DIR / model_dir / 'config.pth', pretrained=False,
                            output_dim=output_dim)
    results = []
    for fold in range(num_fold):
        model.load_state_dict(torch.load(Path('../input') / model_dir / ckpt_path[fold]))

        results_n = {
            'id': [],
            'token_mask': [],
            'offset_mapping': [],
            'probability': [],
        }
        model.eval()
        model.to(device)

        for inputs in tqdm(te_loader):
            input_ids = inputs['input_ids'].to(device)
            token_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            with torch.no_grad():
                y_preds = model(input_ids, token_mask, token_type_ids).sigmoid().cpu().numpy()
                if y_preds.shape[1] > max_len:
                    y_preds[:, :max_len]
                else:
                    y_preds = np.pad(y_preds, ((0, 0), (0, max_len - y_preds.shape[1]), (0, 0)), 'constant', constant_values=0)
                results_n['probability'].append(y_preds[:, :, :1])
                if fold == 0:
                    results_n['offset_mapping'] += [x for x in inputs['offset_mapping']]

        torch.cuda.empty_cache()

        # -------
        print('')
        if fold == 0:
            results.append({
                'probability': np.concatenate(results_n['probability']),
                'offset_mapping': np.array(results_n['offset_mapping'], object)
            })
        else:
            results.append({
                'probability': np.concatenate(results_n['probability'])
            })
        del y_preds, results_n

    # ----
    del model
    gc.collect()
    torch.cuda.empty_cache()

    agg_results = []
    num_samples = len(results[0]['probability'])
    for i in range(num_samples):
        preds = []
        for n in range(num_fold):
            preds.append(results[n]['probability'][i])
        pred = np.mean(preds, axis=0)
        agg_results.append((pred.squeeze(), (results[0]['offset_mapping'][i])))

    return agg_results


def process_feature_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text

# ====================================================
# Main
# ====================================================

INPUT_DIR = Path('../input')
NUM_JOBS = 4

###
# 設定
###
DEBUG = False  # DEBUGだと、それぞれのモデルで2foldのみ

# MODEL_WEIGHTS = [1.0, 0.83571, 0.69586, 0.79665, 0.81276, 0.71487]  # どの割合で足し合わせるか設定。足して1にする必要はない。
MODEL_WEIGHTS = [1.0, 0.83571, 0.69586, 0.79665, 0.81276, 0.71487]  # どの割合で足し合わせるか設定。足して1にする必要はない。
#                 v3,  v2,     v1,     , v2,      v1,      v1
USE_THRESHOLD_LIST = None
if USE_THRESHOLD_LIST:
    case_num_weights = {
    }

"""
config1 = {'model_dir': 'nbme-local-107',  # LB 0.893
      'max_len': 350,
      'batch_size': 80,
      'model_name': 'microsoft/deberta-v3-large',
      'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
      'ckpt_path': [# 'model_0.bin',
                    'model_1.bin',
                    'model_2.bin',
                    'model_3.bin',
                    'model_4.bin',
                    'model_all.bin',
                   ],
       'output_dim': 1  # 最終層の出力
      }
"""
config1 = {'model_dir': 'nbme-local-110',
           'max_len': 350,
           'batch_size': 80,
           'model_name': 'microsoft/deberta-v3-large',
           'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
           'ckpt_path': [
               'model_0.bin',
               'model_1.bin',
               'model_2.bin',
               # 'model_3.bin',
               'model_4.bin',
               'model_all.bin',
           ],
           'output_dim': 1  # 最終層の出力
           }

config2 = {'model_dir': 'nbme-local-092',  # LB 0.893
           'max_len': 355,
           'batch_size': 60,
           'model_name': 'microsoft/deberta-v2-xlarge',
           'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
           'ckpt_path': ['model_0.bin',
                         # 'model_1.bin',
                         # 'model_2.bin',
                         'model_3.bin',
                         'model_4.bin',
                         'model_all.bin'

                         ],
           'output_dim': 1  # 最終層の出力
           }
# そのまま4model

config3 = {'model_dir': 'nbme-local-106',  # LB: 0.892
           'max_len': 500,
           'batch_size': 60,
           'model_name': 'microsoft/deberta-large',
           'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
           'ckpt_path': ['model_0.bin',
                         'model_1.bin',
                         # 'model_2.bin',
                         # 'model_3.bin',
                         # 'model_4.bin',
                         'model_all.bin'
                         ],
           'output_dim': 1  # 最終層の出力
           }
# そのまま3model


config4 = {'model_dir': 'nbme-local-102',  # LB 0.892
           'max_len': 355,
           'batch_size': 60,
           'model_name': 'microsoft/deberta-v2-xlarge',
           'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
           'ckpt_path': ['model_0.bin',
                         'model_1.bin',
                         'model_2.bin',
                         # 'model_3.bin',
                         # 'model_4.bin',
                         'model_all.bin'

                         ],
           'output_dim': 3  # 最終層の出力
           }
# そのまま4モデル

"""
config5 = {'model_dir': 'nbme-local-101',
          'max_len': 500,
          'batch_size': 60,
          'model_name': 'microsoft/deberta-large',
          'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
          'ckpt_path': ['model_0.bin',
                        'model_1.bin',
                        'model_2.bin',
                        'model_3.bin',
                        # 'model_4.bin',
                        # 'model_all.bin'
                       ],
           'output_dim': 3  # 最終層の出力
          }
# そのまま4
"""
config5 = {'model_dir': 'nbme-local-109',
           'max_len': 500,
           'batch_size': 60,
           'model_name': 'microsoft/deberta-large',
           'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
           'ckpt_path': ['model_2.bin',
                         'model_3.bin',
                         'model_4.bin',
                         'model_all.bin',
                         ],
           'output_dim': 1  # 最終層の出力
           }
# 4モデル


"""
config6 = {'model_dir': 'nbme-local-104',  # LB
          'max_len': 355,
          'batch_size': 60,
          'model_name': 'microsoft/deberta-v2-xlarge',
          'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
          'ckpt_path': [# 'model_0.bin',
                        'model_1.bin',
                        'model_2.bin',
                        # 'model_3.bin',
                        # 'model_4.bin',
                        'model_all.bin'

                       ],
           'output_dim': 1  # 最終層の出力
          }
"""
config6 = {'model_dir': 'nbme-local-108',
           'max_len': 355,
           'batch_size': 60,
           'model_name': 'microsoft/deberta-v2-xlarge',
           'preprocess_feature': False,  # process_feature_textをしないならFalse、するならTrue
           'ckpt_path': [

               'model_all.bin',
               'model_1.bin',
               'model_0.bin',

           ],
           'output_dim': 1  # 最終層の出力
           }
# 上書き


# config_listにconfigを入れておく
config_list = [config1, config2, config3, config4, config5, config6]

###
# 設定ここまで
###

assert len(config_list) == len(MODEL_WEIGHTS), 'MODEL_WEIGHTS length mismatch'

if DEBUG:
    for i in range(len(config_list)):
        config_list[i]['ckpt_path'] = config_list[i]['ckpt_path'][:2]

max_lengths = []
for i in range(len(config_list)):
    max_lengths.append(config_list[i]['max_len'])

for num, config in enumerate(config_list):
    model_dir = config['model_dir']
    max_len = config['max_len']
    batch_size = config['batch_size']
    model_name = config['model_name']
    #ckpt_path：用于保存节点信息的路径
    ckpt_path = config['ckpt_path']
    num_fold = len(ckpt_path)
    preprocess_feature = config['preprocess_feature']
    output_dim = config['output_dim']

    tokenizer = get_tokenizer(model_dir)
    test = pd.read_csv(INPUT_DIR / '../input/nbme-score-clinical-patient-notes/test.csv')
    test = df_merge(test, mode='test', preprocess_feature=preprocess_feature)
    print(test)
    test_dataset = TestDataset(tokenizer, test, max_len, return_label=False)

    collate = Collate(tokenizer=tokenizer, return_label=False)
    loader_params = {'batch_size': batch_size,
                     'num_workers': NUM_JOBS,
                     'pin_memory': False,
                     }
    te_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, collate_fn=collate, **loader_params)
    results = make_oof(device, model_dir, max_len, num_fold, ckpt_path, output_dim)
    with open(OUTPUT_DIR / f'oof_word_predict_{num}.pickle', 'wb') as f:
        pickle.dump(results, f)
    del results
    gc.collect()


oof_word_predict_list = []
for num in range(len(config_list)):
    with open(OUTPUT_DIR / f'oof_word_predict_{num}.pickle', 'rb') as f:
        results = pickle.load(f)
    oof_word_predict_list.append(results)

char_probs = []
for i in range(len(test)):
    pn_history = test.loc[i, 'pn_history']
    character_level_probs = []

    for num in range(len(config_list)):
        token_to_text_probability = np.full((len(pn_history)), 0, np.float32)
        p, offset_mapping = oof_word_predict_list[num][i]
        end_prev = 0
        for t, (start, end) in enumerate(offset_mapping):
            if t == max_lengths[num] - 1:
                break
            elif end_prev > 0 and start == 0 and end == 0:
                break
            else:
                token_to_text_probability[end_prev:end] += p[t]
                end_prev = end
        character_level_probs.append(token_to_text_probability)

    if len(MODEL_WEIGHTS) >= 2:
        pred = np.average(character_level_probs, axis=0, weights=MODEL_WEIGHTS)
    else:
        pred = character_level_probs[0]
    if i < 5:
        print(np.max(pred))
    char_probs.append({'pn_history': pn_history, 'character_level_probs': pred})

if not USE_THRESHOLD_LIST:
    result1 = get_results(char_probs, threshold=0.5)

else:
    print(case_num_weights)
    threshold_list = test['case_num'].map(case_num_weights).values
    print(threshold_list[:5])
    result1 = get_results_with_threshold_list(char_probs, threshold_list)

submission = pd.read_csv('../input/nbme-score-clinical-patient-notes/sample_submission.csv')
submission['location'] = result1
print(submission.head())
submission[['id', 'location']].to_csv('submission.csv', index=False)


