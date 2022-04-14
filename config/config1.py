# ====================================================
# CFG：Config
# ====================================================
class CFG:
    #wandb = False
    competition = 'NBME'
    _wandb_kernel = 'nakama'
    debug = False
    #混合精度
    apex = True
    print_freq = 1
    num_workers = 4
    model = "microsoft/deberta-v3-large"
    #余弦学习速率调度器
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 5
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)#原文用的是0.9,0.98
    batch_size = 4
    fc_dropout = 0.2
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 5
    trn_fold = [0,1,2,3,4]
    train = True

#debug模式设置轮次为2
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

# ====================================================
# wandb
# ====================================================
# if CFG.wandb:
#
#     import wandb
#
#     try:
#         from kaggle_secrets import UserSecretsClient
#
#         user_secrets = UserSecretsClient()
#         secret_value_0 = user_secrets.get_secret("wandb_api")
#         wandb.login(key=secret_value_0)
#         anony = None
#     except:
#         anony = "must"
#         print(
#             'If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
#
#
#     def class2dict(f):
#         return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))
#
#
#     run = wandb.init(project='NBME-Public',
#                      name=CFG.model,
#                      config=class2dict(CFG),
#                      group=CFG.model,
#                      job_type="train",
#                      anonymous=anony)

#ALBERT  事件抽取属性抽取