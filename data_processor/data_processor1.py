# from config.config1 import *


from utils.utils1 import get_logger
# from main1 import CFG
import config.config1 as Config
CFG = Config.CFG()
LOGGER = get_logger()

import ast
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
from torch.utils.data import DataLoader, Dataset

# os.system('pip uninstall -y transformers')
# os.system('python -m pip install --no-index --find-links=../input/nbme-pip-wheels transformers')



def Data_Prepare():
    # ====================================================
    # Data Loading
    # ====================================================
    #id; case_num; pn_num ; feature_num; annotation; location
    train = pd.read_csv('../datasets/nbme-score-clinical-patient-notes/train.csv')
    # 转换数据类型
    train['annotation'] = train['annotation'].apply(ast.literal_eval)
    train['location'] = train['location'].apply(ast.literal_eval)
    #feature_num; case_num ; feature_text
    features = pd.read_csv('../datasets/nbme-score-clinical-patient-notes/features.csv')
    #l-year-ago换成1-year-ago
    def preprocess_features(features):
        features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
        return features
    features = preprocess_features(features)
    #pn_num; case_num; pn-history
    patient_notes = pd.read_csv('../datasets/nbme-score-clinical-patient-notes/patient_notes.csv')

    print(f"train.shape: {train.shape}")
    #display(train.head())
    print(f"features.shape: {features.shape}")
    #display(features.head())
    print(f"patient_notes.shape: {patient_notes.shape}")
    #display(patient_notes.head())

    #加入feature_text
    train = train.merge(features, on=['feature_num', 'case_num'], how='left')
    #加入pn-history
    train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')


    # incorrect annotation
    train.loc[338, 'annotation'] = ast.literal_eval('[["father heart attack"]]')
    train.loc[338, 'location'] = ast.literal_eval('[["764 783"]]')

    train.loc[621, 'annotation'] = ast.literal_eval('[["for the last 2-3 months"]]')
    train.loc[621, 'location'] = ast.literal_eval('[["77 100"]]')

    train.loc[655, 'annotation'] = ast.literal_eval('[["no heat intolerance"], ["no cold intolerance"]]')
    train.loc[655, 'location'] = ast.literal_eval('[["285 292;301 312"], ["285 287;296 312"]]')

    train.loc[1262, 'annotation'] = ast.literal_eval('[["mother thyroid problem"]]')
    train.loc[1262, 'location'] = ast.literal_eval('[["551 557;565 580"]]')

    train.loc[1265, 'annotation'] = ast.literal_eval('[[\'felt like he was going to "pass out"\']]')
    train.loc[1265, 'location'] = ast.literal_eval('[["131 135;181 212"]]')

    train.loc[1396, 'annotation'] = ast.literal_eval('[["stool , with no blood"]]')
    train.loc[1396, 'location'] = ast.literal_eval('[["259 280"]]')

    train.loc[1591, 'annotation'] = ast.literal_eval('[["diarrhoe non blooody"]]')
    train.loc[1591, 'location'] = ast.literal_eval('[["176 184;201 212"]]')

    train.loc[1615, 'annotation'] = ast.literal_eval('[["diarrhea for last 2-3 days"]]')
    train.loc[1615, 'location'] = ast.literal_eval('[["249 257;271 288"]]')

    train.loc[1664, 'annotation'] = ast.literal_eval('[["no vaginal discharge"]]')
    train.loc[1664, 'location'] = ast.literal_eval('[["822 824;907 924"]]')

    train.loc[1714, 'annotation'] = ast.literal_eval('[["started about 8-10 hours ago"]]')
    train.loc[1714, 'location'] = ast.literal_eval('[["101 129"]]')

    train.loc[1929, 'annotation'] = ast.literal_eval('[["no blood in the stool"]]')
    train.loc[1929, 'location'] = ast.literal_eval('[["531 539;549 561"]]')

    train.loc[2134, 'annotation'] = ast.literal_eval('[["last sexually active 9 months ago"]]')
    train.loc[2134, 'location'] = ast.literal_eval('[["540 560;581 593"]]')

    train.loc[2191, 'annotation'] = ast.literal_eval('[["right lower quadrant pain"]]')
    train.loc[2191, 'location'] = ast.literal_eval('[["32 57"]]')

    train.loc[2553, 'annotation'] = ast.literal_eval('[["diarrhoea no blood"]]')
    train.loc[2553, 'location'] = ast.literal_eval('[["308 317;376 384"]]')

    train.loc[3124, 'annotation'] = ast.literal_eval('[["sweating"]]')
    train.loc[3124, 'location'] = ast.literal_eval('[["549 557"]]')

    train.loc[3858, 'annotation'] = ast.literal_eval('[["previously as regular"], ["previously eveyr 28-29 days"], ["previously lasting 5 days"], ["previously regular flow"]]')
    train.loc[3858, 'location'] = ast.literal_eval('[["102 123"], ["102 112;125 141"], ["102 112;143 157"], ["102 112;159 171"]]')

    train.loc[4373, 'annotation'] = ast.literal_eval('[["for 2 months"]]')
    train.loc[4373, 'location'] = ast.literal_eval('[["33 45"]]')

    train.loc[4763, 'annotation'] = ast.literal_eval('[["35 year old"]]')
    train.loc[4763, 'location'] = ast.literal_eval('[["5 16"]]')

    train.loc[4782, 'annotation'] = ast.literal_eval('[["darker brown stools"]]')
    train.loc[4782, 'location'] = ast.literal_eval('[["175 194"]]')

    train.loc[4908, 'annotation'] = ast.literal_eval('[["uncle with peptic ulcer"]]')
    train.loc[4908, 'location'] = ast.literal_eval('[["700 723"]]')

    train.loc[6016, 'annotation'] = ast.literal_eval('[["difficulty falling asleep"]]')
    train.loc[6016, 'location'] = ast.literal_eval('[["225 250"]]')

    train.loc[6192, 'annotation'] = ast.literal_eval('[["helps to take care of aging mother and in-laws"]]')
    train.loc[6192, 'location'] = ast.literal_eval('[["197 218;236 260"]]')

    train.loc[6380, 'annotation'] = ast.literal_eval('[["No hair changes"], ["No skin changes"], ["No GI changes"], ["No palpitations"], ["No excessive sweating"]]')
    train.loc[6380, 'location'] = ast.literal_eval('[["480 482;507 519"], ["480 482;499 503;512 519"], ["480 482;521 531"], ["480 482;533 545"], ["480 482;564 582"]]')

    train.loc[6562, 'annotation'] = ast.literal_eval('[["stressed due to taking care of her mother"], ["stressed due to taking care of husbands parents"]]')
    train.loc[6562, 'location'] = ast.literal_eval('[["290 320;327 337"], ["290 320;342 358"]]')

    train.loc[6862, 'annotation'] = ast.literal_eval('[["stressor taking care of many sick family members"]]')
    train.loc[6862, 'location'] = ast.literal_eval('[["288 296;324 363"]]')

    train.loc[7022, 'annotation'] = ast.literal_eval('[["heart started racing and felt numbness for the 1st time in her finger tips"]]')
    train.loc[7022, 'location'] = ast.literal_eval('[["108 182"]]')

    train.loc[7422, 'annotation'] = ast.literal_eval('[["first started 5 yrs"]]')
    train.loc[7422, 'location'] = ast.literal_eval('[["102 121"]]')

    train.loc[8876, 'annotation'] = ast.literal_eval('[["No shortness of breath"]]')
    train.loc[8876, 'location'] = ast.literal_eval('[["481 483;533 552"]]')

    train.loc[9027, 'annotation'] = ast.literal_eval('[["recent URI"], ["nasal stuffines, rhinorrhea, for 3-4 days"]]')
    train.loc[9027, 'location'] = ast.literal_eval('[["92 102"], ["123 164"]]')

    train.loc[9938, 'annotation'] = ast.literal_eval('[["irregularity with her cycles"], ["heavier bleeding"], ["changes her pad every couple hours"]]')
    train.loc[9938, 'location'] = ast.literal_eval('[["89 117"], ["122 138"], ["368 402"]]')

    train.loc[9973, 'annotation'] = ast.literal_eval('[["gaining 10-15 lbs"]]')
    train.loc[9973, 'location'] = ast.literal_eval('[["344 361"]]')

    train.loc[10513, 'annotation'] = ast.literal_eval('[["weight gain"], ["gain of 10-16lbs"]]')
    train.loc[10513, 'location'] = ast.literal_eval('[["600 611"], ["607 623"]]')

    train.loc[11551, 'annotation'] = ast.literal_eval('[["seeing her son knows are not real"]]')
    train.loc[11551, 'location'] = ast.literal_eval('[["386 400;443 461"]]')

    train.loc[11677, 'annotation'] = ast.literal_eval('[["saw him once in the kitchen after he died"]]')
    train.loc[11677, 'location'] = ast.literal_eval('[["160 201"]]')

    train.loc[12124, 'annotation'] = ast.literal_eval('[["tried Ambien but it didnt work"]]')
    train.loc[12124, 'location'] = ast.literal_eval('[["325 337;349 366"]]')

    train.loc[12279, 'annotation'] = ast.literal_eval('[["heard what she described as a party later than evening these things did not actually happen"]]')
    train.loc[12279, 'location'] = ast.literal_eval('[["405 459;488 524"]]')

    train.loc[12289, 'annotation'] = ast.literal_eval('[["experienced seeing her son at the kitchen table these things did not actually happen"]]')
    train.loc[12289, 'location'] = ast.literal_eval('[["353 400;488 524"]]')

    train.loc[13238, 'annotation'] = ast.literal_eval('[["SCRACHY THROAT"], ["RUNNY NOSE"]]')
    train.loc[13238, 'location'] = ast.literal_eval('[["293 307"], ["321 331"]]')

    train.loc[13297, 'annotation'] = ast.literal_eval('[["without improvement when taking tylenol"], ["without improvement when taking ibuprofen"]]')
    train.loc[13297, 'location'] = ast.literal_eval('[["182 221"], ["182 213;225 234"]]')

    train.loc[13299, 'annotation'] = ast.literal_eval('[["yesterday"], ["yesterday"]]')
    train.loc[13299, 'location'] = ast.literal_eval('[["79 88"], ["409 418"]]')

    train.loc[13845, 'annotation'] = ast.literal_eval('[["headache global"], ["headache throughout her head"]]')
    train.loc[13845, 'location'] = ast.literal_eval('[["86 94;230 236"], ["86 94;237 256"]]')

    train.loc[14083, 'annotation'] = ast.literal_eval('[["headache generalized in her head"]]')
    train.loc[14083, 'location'] = ast.literal_eval('[["56 64;156 179"]]')

    #annotation涉及几段
    train['annotation_length'] = train['annotation'].apply(len)

    #display(train['annotation_length'].value_counts())

    # ====================================================
    # CV split
    # ====================================================

    #实例化k折
    Fold = GroupKFold(n_splits=CFG.n_fold)
    #按照pn_num分组
    groups = train['pn_num'].values

    #train['location']代表y，groups表示同一个pn_num的数据不会同时分到训练集测试集
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    #display(train.groupby('fold').size())

    if CFG.debug:
        #display(train.groupby('fold').size())
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)
        #display(train.groupby('fold').size())


    # ====================================================
    # tokenizer
    # ====================================================
    # tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    # tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    # CFG.tokenizer = tokenizer


    from transformers.models.deberta_v2 import DebertaV2TokenizerFast

    #"microsoft/deberta-v3-large"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG.model)
    """
    PreTrainedTokenizerFast(name_or_path='microsoft/deberta-v3-large', vocab_size=128000, 
    model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='right', 
    truncation_side='right', special_tokens={'bos_token': '[CLS]', 'eos_token': '[SEP]', 
    'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 
    'cls_token': '[CLS]', 'mask_token': '[MASK]'})
    """
    CFG.tokenizer = tokenizer

    # ====================================================
    # Define max_len
    # ====================================================
    #统计pn_history和feature_text的长度
    for text_col in ['pn_history']:
        pn_history_lengths = []
        tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            pn_history_lengths.append(length)
        LOGGER.info(f'{text_col} max(lengths): {max(pn_history_lengths)}')

    for text_col in ['feature_text']:
        features_lengths = []
        tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            features_lengths.append(length)
        LOGGER.info(f'{text_col} max(lengths): {max(features_lengths)}')
    #354
    CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3 # cls & sep & sep
    LOGGER.info(f"max_len: {CFG.max_len}")
    return train,CFG.tokenizer

# ====================================================
# Dataset
# ====================================================
#输入self.cfg, self.pn_historys[item], self.feature_texts[item]
def prepare_input(cfg, text, feature_text):
    #inputs:{'input_ids':[],'tokentype_ids':[0，1],'attention_mask':[354维数据中判断有无单词]},均为354维
    inputs = cfg.tokenizer(text, feature_text,
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    #将inputs，tokentype_ids，attention_mask转化为tensor
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

#输入self.cfg,self.pn_historys[item],self.annotation_lengths[item],self.locations[item]
#return_offsets_mapping=False  根据 输入的句子得到 某个词在句子中的位置
def create_label(cfg, text, annotation_length, location_list):
    encoded = cfg.tokenizer(text,
                            add_special_tokens=True,
                            max_length=CFG.max_len,
                            padding="max_length",
                            return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    #需要忽略的index（开始符或padding的）
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    #text出现的地方设为0，否则为-1
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -1
    #self.annotation_lengths如果不为0
    if annotation_length != 0:
        for location in location_list:
            #遍历所有location对
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
    #label:text出现的地方设为0,否则为-1，有答案的地方设为1。label是354个元素的一维张量
    return torch.tensor(label, dtype=torch.float)


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        #输出input_ids,token_type_ids,attention_ids
        inputs = prepare_input(self.cfg,
                               self.pn_historys[item],
                               self.feature_texts[item])
        #354个元素的一维张量，答案位置为1，非答案位置为0，第一个字符和其他padding为-1
        label = create_label(self.cfg,
                             self.pn_historys[item],
                             self.annotation_lengths[item],
                             self.locations[item])
        return inputs, label