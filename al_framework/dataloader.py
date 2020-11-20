import os, json, cv2, itertools, numpy as np
from imantics import Mask
from al_framework.utils import get_ids
from al_framework.logger import save_instances_per_class
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
logger = setup_logger()

# Loads the apollo dataset into detectron2's standard format
def get_apollo_dicts(img_dir, img_ids):
    img_dir = img_dir.split('/')[:-2]
    img_dir = '/'.join(img_dir)

    rgb_dir = img_dir + '/RGB'
    obj_dir = img_dir + '/OBJ'
    seg_dir = img_dir + '/SEG'
    enc_dir = img_dir + '/ENC'
    npy_dir = img_dir + '/NPY'

    skip_ids, classes = [], []
    with open(os.path.join(img_dir, 'skip_ids.txt')) as f:
        for line in f.readlines():
            skip_ids.append(str(line.strip()).zfill(7))
    with open(os.path.join(img_dir, 'apollo_classes_things.txt')) as f:
        for line in f.readlines():
            classes.append(line.strip())

    # progress = 0
    # total_progress = len([x for x in img_ids if x not in skip_ids])
    imgs_anns = []
    for file in next(os.walk(obj_dir))[-1]:
        file_id = file.split('.')[0]
        if (file_id in img_ids) and (file_id not in skip_ids):
            # progress += 1
            anns, segs = [], []
            with open(os.path.join(obj_dir, file)) as f:
                for line in f.readlines():
                    line = line.strip().split()[2:10]
                    if line[0] in classes:
                        anns.append([line[0], list(map(int, list(map(float, line[4:]))))])
            seg_path = os.path.join(npy_dir, file_id+'.npy')
            # print('File', progress, 'out of', total_progress)
            if os.path.exists(seg_path):
                segs = np.load(seg_path, allow_pickle=True)
            else:
                print('npy does not exist', file_id)
                with open(os.path.join(enc_dir, 'Encoding'+file)) as f:
                    for line in f.readlines()[1:]:
                        line = line.strip().split()
                        class_name = line[0].split(':')[0]
                        if class_name in classes:
                            rgb = list(map(int, line[1:]))
                            seg_image = cv2.imread(os.path.join(seg_dir, file_id+'.png'))
                            seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                            this_mask = cv2.inRange(seg_image, np.asarray(rgb), np.asarray([rgb[0], rgb[1], rgb[2]+1]))
                            if cv2.countNonZero(this_mask.reshape(-1)) != 0:
                                print("Found mask")
                                this_mask = this_mask.clip(max=1)
                                polygons = Mask(this_mask).polygons()
                                class_seg = [class_name, polygons.segmentation]
                                segs.append(class_seg)
                np.save(seg_path, np.array(segs, dtype=object))
            imgs_anns.append([file_id+'.jpg', anns, segs])

    dataset_dicts = []
    for img in imgs_anns:
        record = {}

        filename = os.path.join(rgb_dir, img[0])
        height, width = 1080, 1920
        record["file_name"] = filename
        record["image_id"] = filename
        record["height"] = height
        record["width"] = width
      
        objs = []
        segs = img[2]

        for j in range(len(segs)):
            class_name = segs[j][0]
            polygons = segs[j][1]
            assertFailed = False
            for pol in polygons:
                if not (len(pol) >= 6 and len(pol)%2==0):
                    assertFailed = True
                    break
            if assertFailed:
                continue
            x_low, y_low = 1000000, 1000000
            x_high, y_high = -1000000, -1000000
            for pol in polygons:
                for k in range(0, len(pol)-1, 2):
                    x, y = pol[k], pol[k+1]
                    x_low = x if x < x_low else x_low
                    y_low = y if y < y_low else y_low
                    x_high = x if x > x_high else x_high
                    y_high = y if y > y_high else y_high
            obj = {
                "bbox": [x_low, y_low, x_high, y_high],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes.index(class_name),
                "segmentation": [list(map(float, p)) for p in polygons]
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

# Loads the yymnist dataset into detectron2's standard format
def get_yymnist_dicts(img_dir, img_ids):
    img_dir = img_dir.split('/')[:-1]
    img_dir = '/'.join(img_dir)
    imgs_anns = []
    with open(os.path.join(img_dir, 'labels.txt')) as f:
        for line in f.readlines():
            line = line.strip().split('/')[-1]
            img_name = line.split('.')[0]
            if img_name in img_ids:
                anns = line.split()[1:]
                for i in range(len(anns)):
                    anns[i] = list(map(int, anns[i].split(',')))
                imgs_anns.append([img_name+'.jpg', anns])

    dataset_dicts = []
    for img in imgs_anns:
        record = {}

        filename = os.path.join(img_dir, img[0])
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = filename
        record["height"] = height
        record["width"] = width
      
        objs = []
        for bbox in img[1]:
            obj = {
                "bbox": bbox[:4],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": bbox[-1]
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Loads the ballon dataset into detectron2's standard format
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in imgs_anns.items():
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Loads the waymo dataset into detectron2's standard format
def get_waymo_dicts(img_dir, img_ids):
    img_dir = img_dir.split('/')[:-2]
    img_dir = '/'.join(img_dir)

    rgb_dir = img_dir + '/RGB'
    obj_dir = img_dir + '/OBJ'

    classes = []
    with open(os.path.join(img_dir, 'waymo_classes_things.txt')) as f:
        for line in f.readlines():
            classes.append(line.strip())

    progress = 0
    total_progress = len(img_ids)
    imgs_anns = []
    for file in next(os.walk(obj_dir))[-1]:
        file_id = file.split('.')[0]
        if file_id in img_ids:
            progress += 1
            anns = []
            with open(os.path.join(obj_dir, file)) as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    if line[1] in classes:
                        anns.append([line[1], list(map(int, list(map(float, line[2:]))))])
            # print('File', progress, 'out of', total_progress)
            imgs_anns.append([file_id+'.jpg', anns])

    dataset_dicts = []
    for img in imgs_anns:
        record = {}

        filename = os.path.join(rgb_dir, img[0])
        height, width = 1280, 1920
        record["file_name"] = filename
        record["image_id"] = filename
        record["height"] = height
        record["width"] = width
      
        objs = []
        anns = img[1]

        for j in range(len(anns)):
            class_name = anns[j][0]
            bbox = anns[j][1]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes.index(class_name),
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def get_project_waymo_dicts(img_dir, img_ids):
    img_dir = img_dir.split('/')[:-2]
    img_dir = '/'.join(img_dir)
    rgb_dir = img_dir + '/images'

    classes = ['vehicle', 'person', 'cyclist', 'sign']

    progress = 0
    total_progress = len(img_ids)
    imgs_anns = []

    import json
    with open(os.path.join(img_dir, 'labels.json')) as f:
        data = json.load(f)
        for img in data:
            if str(img['image_id']) in img_ids:
                anns = []
                for bbox in img['bboxes']:
                    anns.append([bbox['label'], [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])]])
                imgs_anns.append([str(img['image_id'])+'.jpg', anns])

    dataset_dicts = []
    for img in imgs_anns:
        record = {}

        filename = os.path.join(rgb_dir, img[0])
        height, width = 960, 1280
        record["file_name"] = filename
        record["image_id"] = filename
        record["height"] = height
        record["width"] = width
      
        objs = []
        anns = img[1]

        for j in range(len(anns)):
            class_name = anns[j][0]
            bbox = anns[j][1]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes.index(class_name),
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def get_naplab_dicts(img_dir, img_ids):
    img_dir = img_dir.split('/')[:-2]
    img_dir = '/'.join(img_dir)
    rgb_dir = img_dir + '/images/'
    print(len(img_ids))

    progress = 0
    total_progress = len(img_ids)
    imgs_anns = []

    for file in next(os.walk(rgb_dir))[-1]:
        file_id = file.split('.')[0]
        if file_id in img_ids:
            progress += 1
            anns = []
            if progress%100 == 0:
                print('File', progress, 'out of', total_progress)
            imgs_anns.append([file_id+'.jpg', anns])

    dataset_dicts = []
    for img in imgs_anns:
        record = {}

        filename = os.path.join(rgb_dir, img[0])
        height, width = 960, 1280
        record["file_name"] = filename
        record["image_id"] = filename
        record["height"] = height
        record["width"] = width

        dataset_dicts.append(record)

    return dataset_dicts

'''
When adding a new dataset, remember to update the code in register_dataset(...).
In addition, update create_init_sets(...) (utils.py) and do_pred(...) (pred.py).
'''
def register_dataset(al_cfg, RUN_DIR, ALI, STRATEGY):
    DatasetCatalog.clear()

    TRAIN_IMAGE_IDS, VAL_IMAGE_IDS, FIXED_TEST_IMAGE_IDS, TEST_IMAGE_IDS = get_ids(al_cfg, RUN_DIR, ALI, STRATEGY)
    logger.info("Set Sizes\tSeed Set: {}\tVal Set: {}\tUnlabeled Set: {}".format(len(TRAIN_IMAGE_IDS), len(VAL_IMAGE_IDS), len(TEST_IMAGE_IDS)))    
    IMG_IDS_DIC = {'train': TRAIN_IMAGE_IDS, 'val': VAL_IMAGE_IDS, 'fixed': FIXED_TEST_IMAGE_IDS, 'test': TEST_IMAGE_IDS}
    
    if al_cfg.DATASET_NAME == "apollo":
        apollo_classes = []
        with open(os.path.join(al_cfg.DATASET_DIR, 'apollo_classes_things.txt')) as f:
            for line in f.readlines():
                apollo_classes.append(line.strip())
                
        DatasetCatalog.register(al_cfg.DATASETS.TRAIN, lambda d='train': get_apollo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.VAL,   lambda d='val':   get_apollo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.FIXED, lambda d='fixed': get_apollo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.TEST,  lambda d='test':  get_apollo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        MetadataCatalog.get(al_cfg.DATASETS.TRAIN).set(thing_classes=apollo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.VAL).set(thing_classes=apollo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.FIXED).set(thing_classes=apollo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.TEST).set(thing_classes=apollo_classes)

        save_instances_per_class(al_cfg, RUN_DIR, ALI, STRATEGY, get_apollo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', 'train'), TRAIN_IMAGE_IDS), apollo_classes)
    elif al_cfg.DATASET_NAME == "yymnist":
        yymnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        DatasetCatalog.register(al_cfg.DATASETS.TRAIN, lambda d='train': get_yymnist_dicts(os.path.join(al_cfg.DATASET_DIR, 'Images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.VAL,   lambda d='val':   get_yymnist_dicts(os.path.join(al_cfg.DATASET_DIR, 'Images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.FIXED, lambda d='fixed': get_yymnist_dicts(os.path.join(al_cfg.DATASET_DIR, 'Images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.TEST,  lambda d='test':  get_yymnist_dicts(os.path.join(al_cfg.DATASET_DIR, 'Images', d), IMG_IDS_DIC[d]))
        MetadataCatalog.get(al_cfg.DATASETS.TRAIN).set(thing_classes=yymnist_classes)
        MetadataCatalog.get(al_cfg.DATASETS.VAL).set(thing_classes=yymnist_classes)
        MetadataCatalog.get(al_cfg.DATASETS.FIXED).set(thing_classes=yymnist_classes)
        MetadataCatalog.get(al_cfg.DATASETS.TEST).set(thing_classes=yymnist_classes)

        save_instances_per_class(al_cfg, RUN_DIR, ALI, STRATEGY, get_yymnist_dicts(os.path.join(al_cfg.DATASET_DIR, 'Images', 'train'), TRAIN_IMAGE_IDS), yymnist_classes)
    elif al_cfg.DATASET_NAME == "waymo":
        waymo_classes = []
        with open(os.path.join(al_cfg.DATASET_DIR, 'waymo_classes_things.txt')) as f:
            for line in f.readlines():
                waymo_classes.append(line.strip())

        DatasetCatalog.register(al_cfg.DATASETS.TRAIN, lambda d='train': get_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.VAL,   lambda d='val':   get_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.FIXED, lambda d='fixed': get_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.TEST,  lambda d='test':  get_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', d), IMG_IDS_DIC[d]))
        MetadataCatalog.get(al_cfg.DATASETS.TRAIN).set(thing_classes=waymo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.VAL).set(thing_classes=waymo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.FIXED).set(thing_classes=waymo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.TEST).set(thing_classes=waymo_classes)

        save_instances_per_class(al_cfg, RUN_DIR, ALI, STRATEGY, get_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'RGB', 'train'), TRAIN_IMAGE_IDS), waymo_classes)
    elif al_cfg.DATASET_NAME == "project_waymo":
        project_waymo_classes = ['vehicle', 'person', 'cyclist', 'sign']

        DatasetCatalog.register(al_cfg.DATASETS.TRAIN, lambda d='train': get_project_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.VAL,   lambda d='val':   get_project_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.FIXED, lambda d='fixed': get_project_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.TEST,  lambda d='test':  get_project_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        MetadataCatalog.get(al_cfg.DATASETS.TRAIN).set(thing_classes=project_waymo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.VAL).set(thing_classes=project_waymo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.FIXED).set(thing_classes=project_waymo_classes)
        MetadataCatalog.get(al_cfg.DATASETS.TEST).set(thing_classes=project_waymo_classes)

        save_instances_per_class(al_cfg, RUN_DIR, ALI, STRATEGY, get_project_waymo_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', 'train'), TRAIN_IMAGE_IDS), project_waymo_classes)
    elif al_cfg.DATASET_NAME == "naplab":
        DatasetCatalog.register(al_cfg.DATASETS.TRAIN, lambda d='train': get_naplab_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.VAL,   lambda d='val':   get_naplab_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.FIXED, lambda d='fixed': get_naplab_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        DatasetCatalog.register(al_cfg.DATASETS.TEST,  lambda d='test':  get_naplab_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', d), IMG_IDS_DIC[d]))
        MetadataCatalog.get(al_cfg.DATASETS.TRAIN).set(thing_classes=[])
        MetadataCatalog.get(al_cfg.DATASETS.VAL).set(thing_classes=[])
        MetadataCatalog.get(al_cfg.DATASETS.FIXED).set(thing_classes=[])
        MetadataCatalog.get(al_cfg.DATASETS.TEST).set(thing_classes=[])

        # save_instances_per_class(al_cfg, RUN_DIR, ALI, STRATEGY, get_naplab_dicts(os.path.join(al_cfg.DATASET_DIR, 'images', 'train'), TRAIN_IMAGE_IDS), [])
    else:
        logger.error("Dataset not registered! Code not implemented for dataset: {}".format(al_cfg.DATASET_NAME))
        raise NotImplementedError
