import os, random
from os.path import exists, join
from detectron2.utils.logger import setup_logger
logger = setup_logger()

def create_folders(al_cfg):
    '''
    This function creates the necessary folders that will contain the
    outputs from the current run.
    '''
    output_dir = al_cfg.OUTPUT_DIR
    os.makedirs(output_dir) if not exists(output_dir) else None
    all_runs = next(os.walk(output_dir))[1]
    all_runs.sort()
    RUN_ID = 0 if not all_runs else int(all_runs[-1][-1])+1
    path = join(output_dir, 'run'+str(RUN_ID))
    os.makedirs(path) if not exists(path) else logger.warning("Unable to create RUN_DIR: {}".format(path))
    return 'run'+str(RUN_ID)

def init_cleanup(al_cfg, RUN_DIR, ALI, qs):
    '''
    This function cleans up the output by removing all unnecessary .pth files.
    Files are being removed from active learning iteration 1 to active learning iteration ALI-1 
    '''
    if ALI > 1:
        outputs = next(os.walk(join(al_cfg.OUTPUT_DIR, RUN_DIR)))[1]
        qs_outputs = [o for o in outputs if qs in o and '0' not in o and str(ALI) not in o]
        for output in qs_outputs:
            if exists(join(al_cfg.OUTPUT_DIR, RUN_DIR, output, 'model_final.pth')):
                logger.info('Removing previously saved model: {}'.format(join(al_cfg.OUTPUT_DIR, RUN_DIR, output, 'model_final.pth')))
                os.remove(join(al_cfg.OUTPUT_DIR, RUN_DIR, output, 'model_final.pth'))

def create_init_sets(al_cfg, RUN_DIR, ALI, STRATEGY):
    '''
    This function creates the initial sets by writing image ids to files.
    These files are then accessed when training, evaluating and predicting.
    '''
    try:
        ALL_IMAGE_IDS = []
        if al_cfg.DATASET_NAME == 'apollo':
            all_ids = next(os.walk(join(al_cfg.DATASET_DIR, 'RGB')))[-1]
            skip_ids = []
            with open(join(al_cfg.DATASET_DIR, 'skip_ids.txt')) as f:
                for line in f.readlines():
                    skip_ids.append(str(line.strip()).zfill(7))
            for img_id in all_ids:
                this_id = img_id.split('.')[0]
                if this_id not in skip_ids:
                    ALL_IMAGE_IDS.append(this_id)
        elif al_cfg.DATASET_NAME == 'yymnist':
            all_ids = next(os.walk(join(al_cfg.DATASET_DIR, 'Images/all')))[-1]
            all_ids.remove('labels.txt')
            for img_id in all_ids:
                ALL_IMAGE_IDS.append(img_id.split('.')[0])
        elif al_cfg.DATASET_NAME == 'waymo':
            all_ids = next(os.walk(join(al_cfg.DATASET_DIR, 'RGB')))[-1]
            for img_id in all_ids:
                this_id = img_id.split('.')[0]
                ALL_IMAGE_IDS.append(this_id)
        elif al_cfg.DATASET_NAME == 'project_waymo':
            all_ids = next(os.walk(join(al_cfg.DATASET_DIR, 'images')))[-1]
            for img_id in all_ids:
                this_id = img_id.split('.')[0]
                ALL_IMAGE_IDS.append(this_id)
        elif al_cfg.DATASET_NAME == 'naplab':
            all_ids = next(os.walk(join(al_cfg.DATASET_DIR, 'images')))[-1]
            for img_id in all_ids:
                this_id = img_id.split('.')[0]
                ALL_IMAGE_IDS.append(this_id)
            print(len(all_ids))
        else:
            logger.error("Dataset not registered! Code not implemented for dataset: {}".format(al_cfg.DATASET_NAME))
            raise NotImplementedError
    except StopIteration:
        logger.error("Dataset not found! Add this dataset to the folder datasets: {}".format(al_cfg.DATASET_NAME))
        raise OSError("Could not find dataset!")

    TRAIN_IMAGE_IDS = []
    if al_cfg.UNBALANCED and al_cfg.DATASET_NAME == 'apollo':
        UNBALANCED_IDS = []
        with open(join(al_cfg.DATASET_DIR, 'unbalanced_sets/Sedan.txt')) as f:
            for line in f.readlines():
                UNBALANCED_IDS.append(line.strip())
        with open(join(al_cfg.DATASET_DIR, 'unbalanced_sets/SUV.txt')) as f:
            for line in f.readlines():
                UNBALANCED_IDS.append(line.strip())
        TRAIN_IMAGE_IDS = random.sample(UNBALANCED_IDS, k=al_cfg.AL.INIT_TRAIN_SIZE)
    else:
        TRAIN_IMAGE_IDS = random.sample(ALL_IMAGE_IDS, k=al_cfg.AL.INIT_TRAIN_SIZE)
    VAL_IMAGE_IDS = random.sample(list(set(ALL_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)), k=al_cfg.AL.VAL_SIZE)
    FIXED_TEST_IMAGE_IDS = random.sample(list(set(ALL_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)-set(VAL_IMAGE_IDS)), k=al_cfg.AL.FIXED_TEST_SIZE)
    TEST_IMAGE_IDS = list(set(ALL_IMAGE_IDS)-set(TRAIN_IMAGE_IDS)-set(VAL_IMAGE_IDS)-set(FIXED_TEST_IMAGE_IDS))

    set_ids(al_cfg, RUN_DIR, ALI, STRATEGY, TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS)

def get_ids(al_cfg, RUN_DIR, ALI, STRATEGY):
    '''
    This function returns the image ids from the current ALI, STRATEGY and RUN
    '''
    TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS = [], [], [], []
    path = join(al_cfg.OUTPUT_DIR, RUN_DIR, 'ids', STRATEGY+str(ALI))
    with open(join(path, 'train.txt')) as f:
        TRAIN_IMAGE_IDS = [l.strip() for l in f.readlines()]
    with open(join(path, 'val.txt')) as f:
        VAL_IMAGE_IDS = [l.strip() for l in f.readlines()]
    with open(join(path, 'fixed_test.txt')) as f:
        FIXED_TEST_IMAGE_IDS = [l.strip() for l in f.readlines()]
    with open(join(path, 'test.txt')) as f:
        TEST_IMAGE_IDS = [l.strip() for l in f.readlines()]
    return TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS

def set_ids(al_cfg, RUN_DIR, ALI, STRATEGY, TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS):
    '''
    This function sets the image ids for the current ALI, STRATEGY and RUN
    '''
    path = join(al_cfg.OUTPUT_DIR, RUN_DIR, 'ids', STRATEGY+str(ALI))
    os.makedirs(path) if not exists(path) else None
    with open(join(path, 'train.txt'), 'w+') as f:
        for ID in TRAIN_IMAGE_IDS:
            f.write(ID+"\n")
    with open(join(path, 'val.txt'), 'w+') as f:
        for ID in VAL_IMAGE_IDS:
            f.write(ID+"\n")
    with open(join(path, 'fixed_test.txt'), 'w+') as f:
        for ID in FIXED_TEST_IMAGE_IDS:
            f.write(ID+"\n")
    with open(join(path, 'test.txt'), 'w+') as f:
        for ID in TEST_IMAGE_IDS:
            f.write(ID+"\n")
