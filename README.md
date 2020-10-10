# active-learning-framework
A pool-based Active Learning Framework for object detection and instance-based segmentation tasks

Setup and tutorial coming soon

Our AL Framework can use new models and datasets as long as they are implemented following **Detectron2**'s documentation, and uncertainty based query strategies can easily be added and implemented as long as they handle the expected input and give an expected output.

The AL Framework runs an experiment using all the provided learners in a sequential fashion where each learner is run for a number of Active Learning Iterations (ALIs). A single ALI consists of: initialization and configuration, training, evaluation, inference and prediction, logging, and clean-up. The figure below illustrates the workflow of our AL Framework.

FIGURE
A user provides a config-file and initializes the framework. Initially, samples are selected randomly, labeled, and added to the training set; labels of samples are obtained from a dataset and not manually annotated. Then, a model is fully trained, evaluated on a test set, and used with Active Learning to select a new set of highly informative samples that are added to the training set. This process is repeated for a number of ALIs.

The user can provide a configuration file to run a custom experiment. The following list is part of our contributions and gives a quick overview of the supported functionality:
- Active Learning: The user can provide the number of ALIs to be run, the number of samples to be selected each ALI, a list of query strategies to be used, size of the initial training, validation (used with early stopping), and test set, and the number of samples to be selected during inference from the unlabeled set.
- Whole Spectrum: The user can choose whether samples should be selected from the whole informativeness spectrum or only from the top of the spectrum (e.g., top highly informative samples). If the whole uncertainty spectrum is selected, the user has to provide a sample ratio, which sets the thresholds.
- Early Stopping: The user can activate early stopping, by providing parameters such as patience, delta, and the evaluation period. With early stopping, the model is trained for much longer, and early stopping is initiated based on the Average Precision on the validation set.
- Data Diversity: The user can ensure data diversity. Highly similar samples are removed from the scored sample set using the similarity metric LPIPS (Cite Zhang **Perceptual Similarity**).
- Model and Weights: The user can select the type of model to be used (e.g., Faster R-CNN or Mask R-CNN) and what weights to be loaded from **Detectron2**'s model zoo, or from a previous experiment.
- Dataset and Outputs: The user can provide which dataset to be used and the output directory for the experiment results and logs. Currently supported datasets: YYMNIST, **Apollo Synthetic**, and **Waymo Open**.

The AL Framework logs and saves several values during each ALI that is used for comparison and illustrative purposes:
- Performance values on the test set following COCO's detection evaluation metrics.
- Total instance count of the training set, and for each class.
- Top 10 samples with high and low informativeness.

Other technical information:
- Software: We use **Detectron2** as the object detection and instance segmentation framework. We use Google Colab quick testing and debugging purposes.
- Hardware: We use two Tesla V100 GPU's located on the NAP Server with 32 GB RAM each.

Models used from **Detectron2**'s model zoo:
- coco-pretrained R50-FPN 3x schedule Faster R-CNN Model for Object Detection (Model ID: 137849458)
- coco-pretrained R50 FPN 3x schedule Mask R-CNN Model for Object Detection and Instance-Based Segmentation (Model ID: 137849600)

FIGURE

This figure illustrates the architecture of the AL Framework. The user provides a configuration file from the configs folder to learn.py to run experiments. The AL engine uses **Detectron2**, and contains training and predictions loops. The AL engine uses different Query Strategies and DataLoaders to perform experiments. All logs and outputs are saved in the run folder. Other utility functions are accessed from utils.py. A **Perceptual Similarity** tool is used for data diversity.

## Important References

### Detectron2
```
@misc{wu2019detectron2,
	title        = {Detectron2},
	author       = {Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick},
	year         = 2019,
	journal      = {GitHub repository},
	publisher    = {GihHub},
	howpublished = {\url{https://github.com/facebookresearch/detectron2}}
}
```

### Apollo Synthetic Dataset
apollo.auto.com
```
@misc{apollo,
	title        = {Apollo Synthetic - Photo-Realistic Dataset for Autonomous Driving},
	author       = {Baidu Apollo team},
	year         = 2019,
	howpublished = {\url{http://apollo.auto/synthetic.html}}
}
```

### Waymo Open Dataset
This repository was made using the Waymo Open Dataset, provided by Waymo LLC under license terms available at waymo.com/open.
```
@misc{waymo_open_dataset,
    key={waymo},
    title = {Waymo Open Dataset: An autonomous driving dataset}, howpublished = {\url{https://www.waymo.com/open}},
    year = {2019}
}
```

### Perceptual Similarity
```
@inproceedings{zhang2018perceptual,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={CVPR},
  year={2018}
}
```
