import os, torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset

def do_train(al_cfg, RUN_DIR, ALI, STRATEGY):
    if ALI == 0 and 'model_final.pth' in al_cfg.MODEL.WEIGHTS:
        checkpoint = torch.load(al_cfg.MODEL.WEIGHTS)
        checkpoint.pop('iteration', None)
        torch.save(checkpoint, al_cfg.MODEL.WEIGHTS)
    elif ALI == 1:
        checkpoint = torch.load(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+"init", "model_final.pth"))
        checkpoint.pop('iteration', None)
        torch.save(checkpoint, os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+"init", "model_final.pth"))
    elif ALI > 0:
        checkpoint = torch.load(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+STRATEGY, "model_final.pth"))
        checkpoint.pop('iteration', None)
        torch.save(checkpoint, os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+STRATEGY, "model_final.pth"))

    cfg = get_cfg()
    cfg.merge_from_file(al_cfg.MODEL.TYPE)
    cfg.DATASETS.TRAIN = (al_cfg.DATASETS.TRAIN,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = al_cfg.DATALOADER.NUM_WORKERS
    if ALI == 0:
        cfg.MODEL.WEIGHTS = al_cfg.MODEL.WEIGHTS
        cfg.SOLVER.MAX_ITER = al_cfg.SOLVER.MAX_ITER_INIT
    elif ALI == 1:
        cfg.MODEL.WEIGHTS = os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+"init", "model_final.pth")
        cfg.SOLVER.MAX_ITER = al_cfg.SOLVER.MAX_ITER
    else:
        cfg.MODEL.WEIGHTS = os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+STRATEGY, "model_final.pth")
        cfg.SOLVER.MAX_ITER = al_cfg.SOLVER.MAX_ITER
    cfg.SOLVER.IMS_PER_BATCH = al_cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = al_cfg.SOLVER.BASE_LR
    cfg.OUTPUT_DIR = os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI)+STRATEGY)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = al_cfg.MODEL.BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = al_cfg.MODEL.NUM_CLASSES
    os.makedirs(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI)+STRATEGY), exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (al_cfg.DATASETS.FIXED,)

    evaluator = DatasetEvaluators([COCOEvaluator(al_cfg.DATASETS.FIXED, cfg, True, cfg.OUTPUT_DIR)])
    data_loader = build_detection_test_loader(cfg, al_cfg.DATASETS.FIXED)
    eval_results = inference_on_dataset(trainer.model, data_loader, evaluator)

    for k, v in eval_results.items():
        eval_line = "ALI-" + str(ALI) + "\t"
        for k, v in v.items():
            eval_line += str(k) + ',' + "{0:.8f}".format(float(v)) + "\t"
        with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'evaluations.txt'), 'a+') as o:
            o.write(eval_line+'\n')
