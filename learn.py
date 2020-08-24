import argparse, os
from al_framework.configs import al_cfg
from al_framework.logger import init_write
from al_framework.utils import create_folders, create_init_sets, init_cleanup
from al_framework.dataloader import register_dataset
from al_framework.engine.pred import do_pred
from al_framework.engine.train import do_train
from al_framework.engine.train_es import do_train_es
from detectron2.utils.logger import setup_logger
logger = setup_logger()
import time, datetime
start_time = time.time()

def get_parser():
    parser = argparse.ArgumentParser(description='Active Learning with Detectron2')
    parser.add_argument(
        "config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    return parser

def main():
    args = get_parser().parse_args()
    al_cfg.merge_from_file(args.config_file)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    RUN_DIR = create_folders(al_cfg)
    # RUN_DIR = 'run'+str(2)

    logger.info("Initial run")
    logger.info("ALI: 0")
    create_init_sets(al_cfg, RUN_DIR, 0, "init")
    init_write(al_cfg, RUN_DIR, "INITIAL")
    register_dataset(al_cfg, RUN_DIR, 0, "init")
    if al_cfg.MODEL.USE_EARLY_STOP:
      do_train_es(al_cfg, RUN_DIR, 0, "init")
    else:
      do_train(al_cfg, RUN_DIR, 0, "init")

    for qs in al_cfg.AL.QUERY_STRATEGIES:
        logger.info("Running query strategy: {}".format(qs))
        init_write(al_cfg, RUN_DIR, qs.upper())
        for ALI in range(1, al_cfg.AL.ALI+1):
            logger.info("ALI: {}".format(ALI))
            do_pred(al_cfg, RUN_DIR, ALI, qs)
            register_dataset(al_cfg, RUN_DIR, ALI, qs)
            if al_cfg.MODEL.USE_EARLY_STOP:
                do_train_es(al_cfg, RUN_DIR, ALI, qs)
            else:
                do_train(al_cfg, RUN_DIR, ALI, qs)
            init_cleanup(al_cfg, RUN_DIR, ALI, qs)

    logger.info("Finished!")
    execution_duration = (time.time() - start_time)
    readable_duration = str(datetime.timedelta(seconds=execution_duration))
    print(readable_duration)
    with open(os.path.join(al_cfg.OUTPUT_DIR, RUN_DIR, 'runtime.txt'), 'a+') as o:
        o.write(readable_duration+'\n')

if __name__ == '__main__':
    main()
