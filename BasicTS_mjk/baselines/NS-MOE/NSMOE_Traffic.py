import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))
from basicts.metrics import masked_mae, masked_mse, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import NSMOE

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'Traffic'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = 96
OUTPUT_LEN = 720
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data

#python experiments/train.py -c baselines/NS-MOE/NSMOE_Traffic.py -g 0

# Model architecture and parameters - NS-MOE (Neural Scaling Mixture of Experts)
MODEL_ARCH = NSMOE
NUM_NODES = 862
MODEL_PARAM = {
    "enc_in": NUM_NODES,                        # num nodes
    "dec_in": NUM_NODES,
    "c_out": NUM_NODES,
    "seq_len": INPUT_LEN,
    "label_len": INPUT_LEN//2,                  # start token length used in decoder
    "pred_len": OUTPUT_LEN,                     # prediction sequence length
    "factor": 3,                                # attn factor
    "d_model": 512,                             # model dimension (matching iTransformer)
    "n_heads": 8,                               # number of attention heads
    "e_layers": 4,                              # num of encoder layers (matching iTransformer Traffic)
    "d_layers": 1,                              # num of decoder layers
    "d_ff": 512,                                # feedforward dimension
    "dropout": 0.1,                             # dropout rate
    "freq": 'h',                                # frequency
    "use_norm": True,                           # use normalization
    "output_attention": False,                  # output attention weights
    "embed": "timeF",                           # [timeF, fixed, learned]
    "activation": "gelu",                       # activation function
    
    # NS-MOE specific parameters
    "num_experts": 8,                           # number of experts in MoE
    "top_k_experts": 2,                         # top-k experts to route to
    "aux_loss_weight": 0.01,                    # weight for auxiliary load balancing loss
    
    # Time features
    "num_time_features": 4,                     # number of used time features
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    "day_of_month_size": 31,
    "day_of_year_size": 366
}
NUM_EPOCHS = 20  # Matching iTransformer Traffic config

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'NS-MOE: Neural Scaling Mixture of Experts for Time Series Forecasting (Traffic)'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3, 4]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings - Matching iTransformer Traffic config
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MSE': masked_mse,
                                'RMSE': masked_rmse,
                                'MAPE': masked_mape
                            })
CFG.METRICS.TARGET = 'MSE'  # Matching iTransformer Traffic config
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    'NSMOE',
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
# Optimizer settings - Matching iTransformer Traffic config
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"  # Matching iTransformer
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,                               # Matching iTransformer Traffic
}
# Learning rate scheduler settings - Matching iTransformer Traffic config
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"  # Matching iTransformer
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [5, 10],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0                            # Matching iTransformer
}
# Train data loader settings - Matching iTransformer Traffic config
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 32                # Matching iTransformer Traffic
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 32

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 32

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters - Matching iTransformer Traffic config
CFG.EVAL.USE_GPU = False # Whether to use GPU for evaluation. Default: True
