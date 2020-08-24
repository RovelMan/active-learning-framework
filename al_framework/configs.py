from yacs.config import CfgNode as CN

al_cfg = CN()

al_cfg.AL = CN()
al_cfg.AL.ALI = 50
al_cfg.AL.QUERY_STRATEGIES = ['sument', 'maxent', 'rand']
al_cfg.AL.INIT_TRAIN_SIZE = 2000
al_cfg.AL.VAL_SIZE = 1
al_cfg.AL.FIXED_TEST_SIZE = 3000
al_cfg.AL.INFERENCE_SIZE = 5000
al_cfg.AL.WIDE_SELECTION = False
al_cfg.AL.BATCH_SIZE = 250
al_cfg.AL.RATIO = [150, 75, 25]
al_cfg.AL.DIVERSE = False

al_cfg.MODEL = CN()
al_cfg.MODEL.BATCH_SIZE_PER_IMAGE = 256
al_cfg.MODEL.SCORE_THRESH_TEST = 0.5
al_cfg.MODEL.NUM_CLASSES = 10
al_cfg.MODEL.USE_DROPOUT = False
al_cfg.MODEL.USE_EARLY_STOP = False
al_cfg.MODEL.TYPE = './detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
al_cfg.MODEL.WEIGHTS = 'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl'

al_cfg.EARLY_STOP = CN()
al_cfg.EARLY_STOP.EVAL_PERIOD = 200
al_cfg.EARLY_STOP.MAX_ITER = 10000
al_cfg.EARLY_STOP.PATIENCE = 4
al_cfg.EARLY_STOP.DELTA = 0.05

al_cfg.DATALOADER = CN()
al_cfg.DATALOADER.NUM_WORKERS = 16

al_cfg.SOLVER = CN()
al_cfg.SOLVER.MAX_ITER_INIT = 2500
al_cfg.SOLVER.MAX_ITER = 500
al_cfg.SOLVER.IMS_PER_BATCH = 8
al_cfg.SOLVER.BASE_LR = 0.001

al_cfg.DATASETS = CN()
al_cfg.DATASETS.TRAIN = "apollo_train"
al_cfg.DATASETS.VAL = "apollo_val"
al_cfg.DATASETS.FIXED = "apollo_fixed"
al_cfg.DATASETS.TEST ="apollo_test"

al_cfg.DATASET_NAME = "apollo"
al_cfg.OUTPUT_DIR = './runs/faster_rcnn_apollo/'
al_cfg.DATASET_DIR = './datasets/apollo/'
al_cfg.UNBALANCED = False
