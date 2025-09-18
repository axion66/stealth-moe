import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import MSNSMOE

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'ETTh1'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL']  # Null value in the dataset
# Model architecture and parameters
MODEL_ARCH = MSNSMOE
MODEL_PARAM = {
    "seq_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "enc_in": 7,  # ETTh1 has 7 features
    "dec_in": 7,
    "c_out": 7,
    "label_len": INPUT_LEN//2,  # start token length used in decoder
    "d_model": 256,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 256,
    "dropout": 0.1,
    "freq": 'h',
    "embed": "timeF",
    "activation": 'gelu',
    "output_attention": False,
    "use_norm": True,
    "num_experts": 4,
    "top_k_experts": 2,
    "aux_loss_weight": 0.01,
    "num_time_features": 4,
    "time_of_day_size": 24,
    "day_of_week_size": 7,
    "day_of_month_size": 31,
    "day_of_year_size": 366
}
NUM_EPOCHS = 100

################################ Optimization ################################
LOSS = masked_mae
METRICS = {
    'MAE': masked_mae,
    'MSE': masked_mse,
}
OPTIMIZER = {
    'TYPE': 'Adam',
    'PARAM': {
        "lr": 0.0005,
        "weight_decay": 0.0001,
    }
}
LR_SCHEDULER = {
    'TYPE': 'MultiStepLR',
    'PARAM': {
        'milestones': [1, 25, 50],
        'gamma': 0.5
    }
}

################################# Trainer ####################################
RUNNER = SimpleTimeSeriesForecastingRunner

################################# Other ######################################
CFG = EasyDict()
# ================= General ================= #
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1                             # Number of GPUs to use (0 for CPU mode)

# ================= Environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= Model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]       # Traffic feature indices
CFG.MODEL.TARGET_FEATURES = [0]              # Prediction target indices

# ================= Dataset ================= #
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

# ================= Training ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = LOSS
CFG.TRAIN.OPTIM = OPTIMIZER
CFG.TRAIN.LR_SCHEDULER = LR_SCHEDULER
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
# ================= Scaler ================= #
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

# ================= Validation ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()

# ================= Test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()

# ================= Evaluation ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [12, 24, 48, 96]
CFG.EVAL.USE_GPU = True
CFG.EVAL.METRICS = METRICS

# ================= Data Loader ================= #
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.SHUFFLE = False

CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.SHUFFLE = False

# ================= Misc ================= #
CFG.METRICS = EasyDict()
CFG.METRICS.TYPE = 'regressor'
CFG.METRICS.TARGET = 'MSE'
CFG.METRICS.FUNCS = METRICS

# Runner
CFG.RUNNER = RUNNER
