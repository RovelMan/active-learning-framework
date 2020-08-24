import os, random, cv2, numpy as np
from al_framework.utils import get_ids, set_ids
import al_framework.dataloader as dl
from al_framework.strategies.bvsb import sum_bvsb, max_bvsb, avg_bvsb
from al_framework.strategies.entropy import sum_entropy, max_entropy, avg_entropy
from al_framework.strategies.dropout import uncertainty_aware_dropout
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
logger = setup_logger()

def getFinalSoftmaxes():
    from detectron2.modeling.roi_heads.fast_rcnn import FINAL_SOFTMAXES
    return FINAL_SOFTMAXES

def get_diverse_set(STRATEGY_IDS, SIM_SCORES):
    # IDS = []
    # for SIM in SIM_SCORES:
    #     for STRAT_ID in STRATEGY_IDS:
    #         if SIM[0] == STRAT_ID[0]+'.jpg':
    #             if STRAT_ID not in IDS:
    #                 IDS.append(STRAT_ID)
    # IDS = sorted(IDS, key=lambda x: x[1], reverse=True)
    NEW_IDS = []
    last_id_index = 0
    for SIM in SIM_SCORES:
        for i in range(len(STRATEGY_IDS)):
            if SIM[0] == STRATEGY_IDS[i][0]+'.jpg':
                if STRATEGY_IDS[i] not in NEW_IDS:
                    last_id_index = i
                    NEW_IDS.append(STRATEGY_IDS[i])
    NEW_IDS += STRATEGY_IDS[last_id_index+1:]
    NEW_IDS = sorted(NEW_IDS, key=lambda x: x[1], reverse=True)
    return NEW_IDS

def do_pred(al_cfg, RUN_DIR, ALI, STRATEGY):
    cfg = get_cfg()
    cfg.merge_from_file(al_cfg.MODEL.TYPE)
    cfg.DATASETS.TRAIN = (al_cfg.DATASETS.TRAIN,)
    cfg.DATASETS.TEST = (al_cfg.DATASETS.FIXED,)
    cfg.DATALOADER.NUM_WORKERS = al_cfg.DATALOADER.NUM_WORKERS
    if ALI == 1:
        cfg.OUTPUT_DIR = os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+'init')
        TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS = get_ids(al_cfg, RUN_DIR, ALI-1, 'init')
    elif ALI > 0:
        cfg.OUTPUT_DIR = os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI-1)+STRATEGY)
        TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS = get_ids(al_cfg, RUN_DIR, ALI-1, STRATEGY)

    cfg.SOLVER.IMS_PER_BATCH = al_cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.BASE_LR = al_cfg.SOLVER.BASE_LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = al_cfg.MODEL.BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = al_cfg.MODEL.NUM_CLASSES
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = al_cfg.MODEL.SCORE_THRESH_TEST
    os.makedirs(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI)+STRATEGY), exist_ok=True)

    logger.info('Start prediction using STRATEGY {}'.format(STRATEGY))
    if STRATEGY == 'dropout':
        cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHeadWithDropout"
    if STRATEGY in ['maxent', 'sument', 'avgent', 'maxbvsb', 'sumbvsb', 'avgbvsb', 'dropout']:
        predictor = DefaultPredictor(cfg)
        dataset_dicts = None
        if al_cfg.DATASET_NAME == 'apollo':
            dataset_dicts = dl.get_apollo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', 'test'), TEST_IMAGE_IDS)
        elif al_cfg.DATASET_NAME == 'yymnist':
            dataset_dicts = dl.get_yymnist_dicts(os.path.join(al_cfg.DATASET_DIR, 'Images', 'test'), TEST_IMAGE_IDS)
        elif al_cfg.DATASET_NAME == 'waymo':
            dataset_dicts = dl.get_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', 'test'), TEST_IMAGE_IDS)
        elif al_cfg.DATASET_NAME == 'project_waymo':
            dataset_dicts = dl.get_project_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', 'test'), TEST_IMAGE_IDS)
        elif al_cfg.DATASET_NAME == 'naplab':
            dataset_dicts = dl.get_naplab_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', 'test'), TEST_IMAGE_IDS)
        else:
            logger.error("Dataset not registered! Code not implemented for dataset: {}".format(al_cfg.DATASET_NAME))
            raise NotImplementedError
        progress = 0
        pred_results = []
        predictions = {}
        samples = []
        if STRATEGY == 'dropout':
            samples = random.sample(dataset_dicts, 750) if len(dataset_dicts) > 750 else dataset_dicts
        else:
            samples = random.sample(dataset_dicts, al_cfg.AL.INFERENCE_SIZE) if len(dataset_dicts) > al_cfg.AL.INFERENCE_SIZE else dataset_dicts
        for d in samples:
            if STRATEGY == 'dropout':
                filename = d["file_name"].split('/')[-1].split('.')[0]
                for inf in range(5):
                    im = cv2.imread(d["file_name"])
                    outputs = predictor(im)
                    progress += 1
                    if progress%100 == 0:
                        print('Predicted images:', progress, 'out of', len(samples)*5)
                    softmaxes = getFinalSoftmaxes()
                    # with open('softmaxes.txt') as f:
                        # for line in f.readlines():
                            # softmaxes.append([float(s) for s in line.strip().split(',')])
                    num_instances = len(outputs['instances'])
                    fields = outputs['instances'].get_fields()
                    instances = []
                    for k in range(num_instances):
                        class_id = fields['pred_classes'][k].item()
                        score = fields['scores'][k].item()
                        bbox = np.asarray(fields['pred_boxes'][k].tensor.tolist(), dtype=np.float32)[0]
                        mask = np.array(fields['pred_masks'][k].tolist(), dtype=bool)
                        # print("\t\tScore:", score, "\tClass:", apollo_classes[class_id])
                        instances.append({'class_id': class_id, 'score': score, 'softmax': softmaxes[k], 'bbox': bbox, 'mask': mask})

                    if not filename in predictions.keys():
                        predictions[filename] = [instances]
                    else:
                        predictions[filename].append(instances)
            else:
                im = cv2.imread(d["file_name"])
                outputs = predictor(im)
                progress += 1
                if progress%500 == 0:
                    print('Predicted images:', progress, 'out of', len(samples))
                softmaxes = getFinalSoftmaxes()
                # with open('softmaxes.txt') as f:
                #     for line in f.readlines():
                #         softmaxes.append(line.strip().split(','))
                filename = d["file_name"].split('/')[-1].split('.')[0]
                # print("SOFTMAXES:", filename, softmaxes)
                pred_results.append([filename, softmaxes])

    score_line = "ALI-" + str(ALI) + "-" + STRATEGY + "\t"
    # 250 images - high 150 - mid 75 - low 25
    if STRATEGY in ['maxent', 'sument', 'avgent', 'maxbvsb', 'sumbvsb', 'avgbvsb', 'dropout']:
        STRATEGY_IDS = []
        if STRATEGY == 'maxent':
            STRATEGY_IDS = max_entropy(al_cfg, pred_results, len(pred_results))
        elif STRATEGY == 'sument':
            STRATEGY_IDS = sum_entropy(al_cfg, pred_results, len(pred_results))
        elif STRATEGY == 'avgent':
            STRATEGY_IDS = avg_entropy(al_cfg, pred_results, len(pred_results))
        elif STRATEGY == 'maxbvsb':
            STRATEGY_IDS = max_bvsb(pred_results, len(pred_results))
        elif STRATEGY == 'sumbvsb':
            STRATEGY_IDS = sum_bvsb(pred_results, len(pred_results))
        elif STRATEGY == 'avgbvsb':
            STRATEGY_IDS = avg_bvsb(pred_results, len(pred_results))
        else:
            STRATEGY_IDS = uncertainty_aware_dropout(predictions, len(predictions))

        end_index = len(STRATEGY_IDS) if STRATEGY == 'dropout' else len(pred_results)
        for S_ID in STRATEGY_IDS:
            if S_ID[1] == -1:
                end_index = STRATEGY_IDS.index(S_ID)
                break

        STRATEGY_IDS = STRATEGY_IDS[:end_index]

        '''Image similarity should be processed here'''
        ''' Currently supports waymo only '''
        if al_cfg.AL.DIVERSE:
            logger.info('Creating a diverse set... Current size: {}'.format(len(STRATEGY_IDS)))
            # create a file(s) containing STRATEGY_IDS which will be used by the similarity_framework
            with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI)+STRATEGY, 'img_ids.txt'), 'w+') as o:
                for S_ID in STRATEGY_IDS:
                    o.write(S_ID[0]+'.jpg\n')
            # create neccessary inputs for similarity_framework
            data_dir = os.path.join(al_cfg.DATASET_DIR, 'RGB')
            out_dir = os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI)+STRATEGY, 'lpips_sim.txt')
            # start similarity_framework using subprocess, maybe use it multiple times?
            import subprocess
            cmd = ['python3',
                'similarity_framework/PerceptualSimilarity/compute_ids_pair_faster_check.py',
                '-d', data_dir, '-o', out_dir, '-b', str(al_cfg.AL.BATCH_SIZE)]
            subprocess.Popen(cmd).wait()
            # read sim scores
            sims_pre, SIM_SCORES = [], []
            with open(out_dir) as f:
                sims_pre = f.read().split('(')[1:]
            for sim in sims_pre:
                sim = sim.split('): ')
                SIM_SCORES.append([sim[0].split(', ')[0], sim[0].split(', ')[1], float(sim[1])])
            # use STRATEGY_IDS and SIM_SCORES to find thresholds
            STRATEGY_IDS = get_diverse_set(STRATEGY_IDS, SIM_SCORES)
            with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, "output"+str(ALI)+STRATEGY, 'img_ids_after.txt'), 'w+') as o:
                for S_ID in STRATEGY_IDS:
                    o.write(S_ID[0]+'.jpg\n')
            logger.info('Finished! Size after diversification: {}'.format(len(STRATEGY_IDS)))
        '''Image similarity should be processed here'''

        if len(STRATEGY_IDS) > al_cfg.AL.BATCH_SIZE:
            for S_ID in STRATEGY_IDS[:10]:
                score_line += S_ID[0] + ',' + "{0:.8f}".format(float(S_ID[1])) + " "
            for S_ID in STRATEGY_IDS[-10:]:
                score_line += S_ID[0] + ',' + "{0:.8f}".format(float(S_ID[1])) + " "

            if al_cfg.AL.WIDE_SELECTION:
                HIGH_IDS = STRATEGY_IDS[:al_cfg.AL.RATIO[0]]
                LOW_IDS = STRATEGY_IDS[-(al_cfg.AL.RATIO[1]+1):]
                MID_IDS = random.sample(STRATEGY_IDS[al_cfg.AL.RATIO[0]:-(al_cfg.AL.RATIO[2]+1)], k=al_cfg.AL.RATIO[1])
                STRATEGY_IDS = HIGH_IDS + MID_IDS + LOW_IDS

                for j in range(len(STRATEGY_IDS)):
                    STRATEGY_IDS[j] = STRATEGY_IDS[j][0]

                TRAIN_IMAGE_IDS += STRATEGY_IDS[:al_cfg.AL.BATCH_SIZE]
                VAL_IMAGE_IDS += STRATEGY_IDS[al_cfg.AL.BATCH_SIZE:]
                TEST_IMAGE_IDS = list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)-set(VAL_IMAGE_IDS))
            else:
                for j in range(len(STRATEGY_IDS)):
                    STRATEGY_IDS[j] = STRATEGY_IDS[j][0]

                TRAIN_IMAGE_IDS += STRATEGY_IDS[:al_cfg.AL.BATCH_SIZE]
                # VAL_IMAGE_IDS += STRATEGY_IDS[al_cfg.AL.BATCH_SIZE:al_cfg.AL.BATCH_SIZE+al_cfg.AL.VAL_SIZE]
                VAL_IMAGE_IDS += [STRATEGY_IDS[al_cfg.AL.BATCH_SIZE+1]]
                TEST_IMAGE_IDS = list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)-set(VAL_IMAGE_IDS))
        else:
            score_line += "Few predictions. Randomly chosen images"
            TRAIN_IMAGE_IDS += random.sample(TEST_IMAGE_IDS, k=al_cfg.AL.BATCH_SIZE)
            # VAL_IMAGE_IDS += random.sample(list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)), k=al_cfg.AL.VAL_SIZE)
            VAL_IMAGE_IDS += random.sample(list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)), k=1)
            TEST_IMAGE_IDS = list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)-set(VAL_IMAGE_IDS))
    else:
        score_line += "Randomly query strategy"
        TRAIN_IMAGE_IDS += random.sample(TEST_IMAGE_IDS, k=al_cfg.AL.BATCH_SIZE)
        # VAL_IMAGE_IDS += random.sample(list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)), k=al_cfg.AL.VAL_SIZE)
        VAL_IMAGE_IDS += random.sample(list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)), k=1)
        TEST_IMAGE_IDS = list(set(TEST_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)-set(VAL_IMAGE_IDS))

    with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'strategy_scores.txt'), 'a+') as o:
        o.write(score_line+'\n')

    set_ids(al_cfg, RUN_DIR, ALI, STRATEGY, TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS)
