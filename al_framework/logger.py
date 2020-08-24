import os, itertools, numpy as np

def save_instances_per_class(al_cfg, RUN_DIR, ALI, STRATEGY, dataset_dicts, class_names):
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    num_images = 0
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]
        num_images += 1

    data = list(itertools.chain(*[[class_names[i], int(v)] for i, v in enumerate(histogram)]))
    total_num_instances = sum(data[1::2])

    instances_line = "ALI-" + str(ALI) + "-" + STRATEGY + "\t"
    instances_line += 'IMAGES,' + str(num_images) + ' '
    instances_line += 'TOTAL,' + str(total_num_instances) + ' '
    for e in range(0, len(data)-1, 2):
        instances_line += data[e] + ',' + str(data[e+1]) + " "
    with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'strategy_instances.txt'), 'a+') as o:
        o.write(instances_line+'\n')

def init_write(al_cfg, RUN_DIR, text):
    '''
    Writes dividers in the output files after each initial run and query strategy
    '''
    with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'evaluations.txt'), 'a+') as o:
        o.write(text+"\n")
    with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'strategy_scores.txt'), 'a+') as o:
        o.write(text+"\n")
    with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'strategy_instances.txt'), 'a+') as o:
        o.write(text+"\n")
