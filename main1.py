import config.config1 as Config
CFG = Config.CFG()
from train1 import train_loop
from data_processor.data_processor1 import Data_Prepare
from utils.utils1 import get_logger
LOGGER = get_logger()
from utils.utils1 import *

"""
目前共训练5轮，每轮2860批，batch_size=4
"""
# ====================================================
# 初始化
# ====================================================

# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
import shutil
from pathlib import Path

#根据transformers的安装地址定
transformers_path = Path("/Users/hann/miniforge3/envs/dl_mac/lib/python3.8/site-packages/transformers")

#
input_dir =Path("/Users/hann/Documents/ju_code/Kaggle_NBME/datasets/deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py', "deberta__init__.py"]:
    if str(filename).startswith("deberta"):
        filepath = deberta_v2_path/str(filename).replace("deberta", "")
    else:
        filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)

#设置文件夹
# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ====================================================
# Library
# ====================================================
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import torch
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#准备数据和词嵌入
"""
一、载入train.csv,根据feature.csv和pn_history.csv为其加入feature_text和pn_history列；加入annotation_length列；以pn_num为组将数据分折，并加入fold列。
若debug模式，则随机选取100个训练样本。
二、导入分词器DebertaV2TokenizerFast，赋值给CFG.tokenizer，并计算最大序列长度。
"""
train,CFG.tokenizer = Data_Prepare()

if __name__ == '__main__':

    def get_result(oof_df):
        #创建标签掩码
        labels = create_labels_for_scoring(oof_df)
        predictions = oof_df[[i for i in range(CFG.max_len)]].values
        char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
        results = get_results(char_probs, th=0.5)
        preds = get_predictions(results)
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')


    if CFG.train:
        oof_df = pd.DataFrame()
        #遍历每折
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                #准备训练测试数据，设置优化器，调度器，损失函数
                """
                输入：训练集，fold
                过程：一、对每一折划分测试、训练集，存入Dataloader;用CustomModel类导入模型，将参数导出至OUTPUT_DIR + 'config.pth'；
                    设置优化器参数，优化器AdamW；设置调度器；设置损失函数：BCEWithLogitsLoss。
                    二、对每一折分epoch放入train_fn开始训练，输出每一批的平均损失。
                    
                    train_fn过程：调用TrainDataset类的__getitem__方法。把pn_history和feature_text转化为：input_ids,token_type_ids,attention_ids
                    把text和annotation转化为label。
                    （input_ids 就是一连串 token 在字典中的对应id。形状为 (batch_size, sequence_length)。
                    token_type_ids 可选。就是 token 对应的句子id，值为0或1（0表示对应的token属于第一句，1表示属于第二句）。形状为(batch_size, sequence_length)。
                    attention_mask 可选。各元素的值为 0 或 1 ，避免在 padding 的 token 上计算 attention（1不进行masked，0则masked）。
                    形状为(batch_size, sequence_length)。）
                   （label形状为（batch_size, sequence_length）text出现的地方设为0,否则为-1，有答案的地方设为1）
                   
                   三、对每一折计算验证损失，验证集预测，以阈值为0.5得到最后的结果，并计算评估指标。
                   四、对每一折，保存评价指标最好的模型。
                   五、输出每一折的验证集的预测prediction。
                """
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                #每一折计算评价指标
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        #对所有折一起计算评价指标
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR + 'oof_df.pkl')

    # if CFG.wandb:
    #     wandb.finish()

"""
1.今天第二次提交的比第一次好的话，就从那个模型再优化一下。"Adam"，dropout，可不可以看一下（周）   epoch，损失函数(包)。
2.试试模型ensemble。
"""


