import os
from yacs.config import CfgNode as CN


# [LIMO] 패키지 루트 (yolopx_train/). dataset/, weights/ 를 여기 기준 상대경로로 찾음
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

_C = CN()

_C.LOG_DIR =  'runs/'
_C.GPUS = (0,)     # 显卡数 = len(GPUS)
_C.WORKERS = 4      # [LIMO] 8 -> 4 (Windows 메모리 부족 방지, 에러 나면 0)
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 10   # [LIMO] 2150 -> 10 (작은 데이터셋)
_C.AUTO_RESUME =False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
# [LIMO] 멀티클래스 헤드 (cvat2yolopx.py 의 classes.json 과 같은 순서)
#   주행가능영역 헤드: 0 배경 / 1 drivable(내 차선) / 2 drivable_left / 3 drivable_right
#   차선 헤드        : 0 배경 / 1 lane / 2 stop_line
#   검출 헤드        : DET_CLASSES 순서가 클래스 id
_C.num_seg_class = 4        # 주행가능영역 헤드 출력 채널 수 (배경 포함)
_C.num_ll_class = 3         # 차선 헤드 출력 채널 수 (배경 포함)
_C.DA_FLIP_SWAP = [2, 3]    # 좌우반전 증강 시 서로 바꿀 주행가능영역 클래스 (왼쪽 <-> 오른쪽)
_C.DET_CLASSES = [
    'traffic_light_red', 'traffic_light_yellow', 'traffic_light_green',
    'sign_pedestrian', 'sign_slow', 'sign_A', 'sign_B', 'sign_parking',
    'cone', 'road_slow_text', 'crosswalk', 'zigzag',
]
# [LIMO] 좌우반전하면 안 되는 검출 클래스: 이 박스가 있는 이미지는 좌우반전을 건너뜀
#   sign_A / sign_B 는 서로 거울상이라, 반전하면 모순된 정답이 생겨 A/B 를 헷갈리게 됨
_C.NO_FLIP_DET_CLASSES = ['sign_A', 'sign_B']

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']

# MODIFY
_C.MODEL.PRETRAINED = ''

#

_C.MODEL.PRETRAINED_DET = ""
# [LIMO] 파인튜닝용 사전학습 가중치: 모델 가중치만 불러오고 epoch 0 부터 학습 (optimizer/epoch 은 무시)
_C.MODEL.FINETUNE = os.path.join(PACKAGE_ROOT, 'weights', 'epoch-195.pth')
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
# [LIMO] [?, det, da_seg, ll_seg, ll_tversky, ...]
#        검출 박스 라벨이 없는 데이터로만 학습할 때는 두 번째 값(det)을 0.0 으로
_C.LOSS.MULTI_HEAD_LAMBDA = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
_C.LOSS.FL_GAMMA = 2.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.2 # lane line iou loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
# [LIMO] yolopx_train/dataset/ 기준 상대경로 (train.py --data-root 로 바꿀 수 있음)
_C.DATASET.DATAROOT = os.path.join(PACKAGE_ROOT, 'dataset', 'images')              # the path of images folder
_C.DATASET.LABELROOT = os.path.join(PACKAGE_ROOT, 'dataset', 'det_annotations')    # the path of det_annotations folder
_C.DATASET.MASKROOT = os.path.join(PACKAGE_ROOT, 'dataset', 'da_seg_annotations')  # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = os.path.join(PACKAGE_ROOT, 'dataset', 'll_seg_annotations')  # the path of ll_seg_annotations folder
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [480, 640]   # [LIMO] LIMO 카메라 640x480 (height, width)

# training data augmentation
_C.DATASET.FLIP = True   # [LIMO] 좌우반전 (반전 시 drivable_left/right 클래스 자동 교환)
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add

# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.1  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100   # [LIMO] 200 -> 100 (train.py --epochs 로 변경 가능)

_C.TRAIN.VAL_FREQ = 5      # [LIMO] 20 -> 5 (검증 + 체크포인트 저장 주기)
_C.TRAIN.BATCH_SIZE_PER_GPU = 8   # [LIMO] 32 -> 8 (GPU 메모리 8~12GB 기준, --batch-size 로 변경)
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

_C.mosaic_rate = 0.0  # 1.0
_C.mixup_rate = 0.15

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task

_C.TRAIN.PLOT = False                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 8    # [LIMO] 32 -> 8
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = False
_C.TEST.NMS_CONF_THRESHOLD  = 0.001
_C.TEST.NMS_IOU_THRESHOLD  = 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
