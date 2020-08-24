import os, torch, logging, numpy as np
from os.path import exists, join
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, inference_on_dataset
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
logger = logging.getLogger("detectron2")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            print('\tNo EarlyStopping')
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('\tNo EarlyStopping')
            self.best_score = score
            self.counter = 0

def removeOldJSON(al_cfg):
    metadata = MetadataCatalog.get(al_cfg.DATASETS.VAL)
    if hasattr(metadata, "json_file"):
        logger.info('Removing previously cached JSON file')
        del metadata.json_file

def do_test(cfg, model):
    dataset_name = cfg.DATASETS.TEST[0]
    evaluator = DatasetEvaluators([COCOEvaluator(dataset_name, cfg, True, cfg.OUTPUT_DIR)])
    data_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, data_loader, evaluator)
    return results

def start_train(al_cfg, cfg, model, resume=False):
    early_stopping = EarlyStopping(patience=al_cfg.EARLY_STOP.PATIENCE, delta=al_cfg.EARLY_STOP.DELTA, verbose=True)
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                results = do_test(cfg, model)
                bbox_results = results['bbox']
                AP = bbox_results['AP']
                comm.synchronize()
                print('AP:', AP, '\tValue:', 1-(AP/100))
                early_stopping(1-(AP/100))
                storage.put_scalars(**bbox_results)    
                if early_stopping.counter < 1:
                    checkpointer.save('model_final')

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            if early_stopping.early_stop:
                print("EARLY STOPPING INITIATED AT ITERATION:", iteration)
                # checkpointer.save('model_final')
                break

def do_train_es(al_cfg, RUN_DIR, ALI, STRATEGY):
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
    cfg.DATASETS.TEST = (al_cfg.DATASETS.VAL,)
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

    cfg.TEST.EVAL_PERIOD = al_cfg.EARLY_STOP.EVAL_PERIOD
    cfg.SOLVER.MAX_ITER = al_cfg.EARLY_STOP.MAX_ITER 

    removeOldJSON(al_cfg)
    model = build_model(cfg)
    start_train(al_cfg, cfg, model)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (al_cfg.DATASETS.FIXED,)

    evaluator = DatasetEvaluators([COCOEvaluator(al_cfg.DATASETS.FIXED, cfg, True, cfg.OUTPUT_DIR)])
    data_loader = build_detection_test_loader(cfg, al_cfg.DATASETS.FIXED)
    eval_results = inference_on_dataset(model, data_loader, evaluator)

    for k, v in eval_results.items():
        eval_line = "ALI-" + str(ALI) + "\t"
        for k, v in v.items():
            eval_line += str(k) + ',' + "{0:.8f}".format(float(v)) + "\t"
        with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'evaluations.txt'), 'a+') as o:
            o.write(eval_line+'\n')
