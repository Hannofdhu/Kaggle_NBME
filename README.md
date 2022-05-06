# Kaggle_NBME
# 文件说明

- ./main1.py表示./model/Deberta-v3-large baseline.py的主函数，其他代码被放置在剩下的文件夹。
- ./model/Deberta-v3-large baseline.py是用Derberta-v3-large做的建模，设置了5折交叉验证。
- ./model/PubMed-base-baseline.py是对Derberta-v3-large脚本做的调整，引入了PubMedBert,同时改变了tokenizer，最终效果在0.72左右。
- ./model/PubMed-second.py是使用另外一套代码运行PubMedBert，分割数据时用的是train_test_split，没有分折训练。
- ./model/PubMed-Third-5-folds.ipynb对PubMed-second.py做了改进，融合了Deberta-v3-large baseline.py的数据处理细节，增加了5折交叉验证，并分别保存模型。在推理的时候进行模型融合。
根据实验，去掉scheduler，使得PubMed-Third-5-folds.ipynb效果变好，且使用普通KFold比GroupKFold效果好。Bert后接的层，试过两层512神经元的MLP，一层直接映射到1的MLP（全连接层均使用ReLu激活），双向双层LSTM、GRU。其中一层直接映射到1的MLP效果最好，其次是LSTM，GRU，两层512神经元的MLP，但它们差别不大。

# PubMed-Third-5-folds
简单说明网络层构建和预测过程：

- 首先，取序列最大长度max_len为416，将pn_history和feature_text输入PubMedBert的分词器，得到两个序列的input_ids，attention_mask和token_type_ids，它们的维度均为（batch_size,max_len）。

- 在前向训练时，分批将上述分词后得到的序列编码输入PubMedBert模型，并提取模型最后一层输出的所有特征，特征的维度为（batch_size,max_len,768），其中768指模型的特征维度。

- 得到模型特征后，将其输入两层线性层，线性层的神经元均为512，其中输入维度为768，输出维度为1；线性层使用ReLU激活函数，并加入Dropout方法，设置p为0.2。最后得到输出特征的维度为（batch_size,max_len）。

- 得到输出特征后，对其进行后处理。首先，将它们输入Sigmoid函数，得到维度为（batch_size,max_len）的概率分布序列；同时，将pn_history输入PubMedBert的分词器，得到病历的offset_mapping，它是列表嵌套元组的形式，表示每个词在段落中索引位置的开头和结尾。

- 遍历概率分布序列，若其中对应某些连续词的元素值大于0.5（本文的阈值，推理时可调节），则判断第一个词的开头索引为开始索引，最后一个词的结束索引为结束索引。由此，可以得到一个或多个连续片段作为反映feature_text的片段。

# 经验教训
- 超参数的存储方式——字典、类、argparse库
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--xxx',default=xxx,action='store_true',type=int,help='')
args = parser.parse_args()

#引用
args.model_dir....
```
- 设置超参数的时候，勿忘设置debug模式，如果开启，在epochs，trn_fold，数据读取上都有缩减。
- bert的各种编码形式(以问答为例)：

input_ids: (context+question+padding（0）+special tokens)的词表索引。

attention_mask：0或1。0表示不需要attention计算，1表示需要attention计算。

token_type_ids：0或1。区分(context,queation)和其他序列(包括padding, special tokens)

Labels: 0或1，context和question为0，其中包含答案的span为1，其他为0（包括padding, special tokens)。

- 分折训练/推理需要注意的点：

1. 每一折都要重新定义dataset和DataLoader。
2. 每一折都要初始化各种参数，包括：train_loss_data、valid_loss_data、score_data_list、valid_loss_min、optimizer等。
3. 每一折都要重新定义model。
```python
model = CustomModel(hyperparameters).to(DEVICE)
```
4. 每一折都要保存最优模型，“最优模型”是根据验证损失确定的。
```python
#保存最优模型
if valid_loss < best_loss:
  best_loss = valid_loss
  torch.save(model.state_dict(), f"nbme_pubmed_bert_fold{fold_num+1}.pth")
```
5. 每一折训练完后记得删除内存、缓存、进行垃圾回收
```python
del model,preds
import gc
torch.cuda.empty_cache()
"""
Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible in nvidia-smi.
empty_cache() doesn’t increase the amount of GPU memory available for PyTorch. However, it may help reduce fragmentation of GPU memory in certain cases
"""
gc.collect()
#若被调用时不包含参数，则启动完全的垃圾回收
```
6. K折融合推理时，先循环fold再循环batch,最后输出概率分布的平均值。
```python
#准备分折预测
predictions = []
for fold in hyperparameters['trn_fold']:
    model = CustomModel(hyperparameters).to(DEVICE)
    model.load_state_dict(torch.load(f"nbme_pubmed_bert_fold{fold}.pth"))
    model.eval()
    #初始化指标容器
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
    
    del model,preds; gc.collect()
    torch.cuda.empty_cache()
#五折总输出的平均
predictions = np.mean(predictions,axis=0)

#输出
location_preds = get_location_predictions(predictions, offsets, seq_ids, test=True)
test_df["location"] = location_preds
test_df[["id", "location"]].to_csv("submission.csv", index = False)
```