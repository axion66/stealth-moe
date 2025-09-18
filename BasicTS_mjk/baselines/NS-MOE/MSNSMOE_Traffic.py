import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
from easydict import EasyDict
# Import from local arch directory
from .arch import MSNSMOE
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.utils import load_adj

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "Multi-Scale NS-MOE model configuration for Traffic dataset"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Traffic"
CFG.DATASET_TYPE = "Traffic"
CFG.DATASET_INPUT_LEN = 96
CFG.DATASET_OUTPUT_LEN = 720
CFG.GPU = 0
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MSNSMOE"
CFG.MODEL.ARCH = MSNSMOE
INPUT_LEN = CFG.DATASET_INPUT_LEN
OUTPUT_LEN = CFG.DATASET_OUTPUT_LEN
CFG.MODEL.PARAM = {
    "seq_len": INPUT_LEN,
    "label_len": INPUT_LEN/2,       # start token length used in decoder
    "pred_len": OUTPUT_LEN,         # prediction sequence length
    "enc_in": 862,                               # encoder input size (number of variables)
    "dec_in": 862,                               # decoder input size
    "c_out": 862,                                # output size 
    "d_model": 512,                             # model dimension (matching iTransformer)
    "n_heads": 8,                               # number of attention heads
    "e_layers": 4,                              # num of encoder layers (matching iTransformer Traffic)
    "d_layers": 1,                              # num of decoder layers
    "d_ff": 512,                                # feedforward dimension
    "dropout": 0.1,                             # dropout rate
    "freq": 'h',                                # frequency
    "use_norm": True,                           # use normalization
    "output_attention": False,                  # output attention weights
    
    # MoE specific parameters
    "num_experts": 8,                           # number of experts in MoE
    "top_k_experts": 2,                         # number of top experts to route to
    "aux_loss_weight": 0.01,                    # auxiliary loss weight for load balancing
}
CFG.MODEL.FROWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = "MSE"
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0001,
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 20
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False

# ================= evaluate ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [12, 24, 48, 96, 192, 288, 336, 720]
CFG.EVAL.USE_GPU = True

# ================= lr_scheduler ================= #
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 10],
    "gamma": 0.5
}

# ================= metrics ================= #
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = ["MAE", "MSE", "RMSE", "MAPE", "WAPE"]
CFG.METRICS.TARGET = "MSE"
CFG.METRICS.NULL_VAL = CFG.NULL_VAL
