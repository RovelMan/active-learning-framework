import torch, numpy as np, torch.nn.functional as F
from torch import nn
import fvcore.nn.weight_init as weight_init
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHeadWithDropout(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        self.fcs.append(nn.Dropout(0.5))

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size

def uncertainty_aware_dropout(predictions, k):
    image_uncertainty = []
    for filename, samples in predictions.items():
        observations = {}
        obs_id = 0
        threshold = 0.5
        for detections in samples:
            for detection in detections:
                if not observations:
                    observations[obs_id] = [detection]
                else:
                    addThis = None
                    for group, ds, in observations.items():
                        for d in ds:
                            thisMask = detection['mask']
                            otherMask = d['mask']
                            overlap = np.logical_and(thisMask, otherMask)
                            union = np.logical_or(thisMask, otherMask)
                            IOU = overlap.sum()/float(union.sum())
                            if IOU <= threshold:
                                break
                            else:
                                addThis = [group, detection]
                                break
                        if addThis:
                            break
                    if addThis:
                        observations[addThis[0]].append(addThis[1])
                    else:
                        obs_id += 1
                        observations[obs_id] = [detection]

        sum_uncertainty = 0.0
        for key, val in observations.items():
            mean_softmax = np.mean(np.array([v['softmax'] for v in val]), axis=0)
            mean_bbox = np.mean(np.array([v['bbox'] for v in val]), axis=0)
            mean_mask = np.mean(np.array([v['mask'].flatten() for v in val]), axis=0)
            mean_mask[mean_mask <= 0.3] = 0.0 # Mask R-CNN Threshold
            # mean_mask = mean_mask.reshape(-1, 1920) #Apollo
            mean_mask = mean_mask.reshape(-1, 1280) #NAPLab

            mask_IOUs = []
            for v in val:
                current_mask = v['mask']
                overlap = np.logical_and(mean_mask, current_mask)
                union = np.logical_or(mean_mask, current_mask)
                if union.sum() > 0:
                    mask_IOUs.append(float(overlap.sum())/float(union.sum()))

            bbox_IOUs = []
            boxAArea = (mean_bbox[2] - mean_bbox[0] + 1) * (mean_bbox[3] - mean_bbox[1] + 1)
            for v in val:
                current_bbox = v['bbox']
                xA = max(mean_bbox[0], current_bbox[0])
                yA = max(mean_bbox[1], current_bbox[1])
                xB = min(mean_bbox[2], current_bbox[2])
                yB = min(mean_bbox[3], current_bbox[3])
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                boxBArea = (current_bbox[2] - current_bbox[0] + 1) * (current_bbox[3] - current_bbox[1] + 1)
                bbox_IOUs.append(interArea / float(boxAArea + boxBArea - interArea))

            u_sem = max(mean_softmax)
            u_spl_m = sum(mask_IOUs)/len(val)
            u_spl_b = sum(bbox_IOUs)/len(val)
            u_n = 0.0
            if len(val) <= 5:
                u_n = len(val)/5
            elif len(val) >= 5*2:
                u_n = 0.0
            else:
                u_n = 1.0-((len(val)%5)/5)

            u_sem_w = u_sem*u_n
            u_spl_m_w = u_spl_m*u_n
            u_spl_b_w = u_spl_b*u_n
            u_h_m_w = u_sem*u_spl_m*u_n
            u_h_m_b_w = u_sem*u_spl_m*u_spl_b*u_n

            if u_h_m_b_w >= 0.2:
                sum_uncertainty += 1.0-(u_h_m_b_w)
                
        if sum_uncertainty == 0.0:
            sum_uncertainty = -1
        image_uncertainty.append([filename, sum_uncertainty])
    TOP_IDS = sorted(image_uncertainty, key=lambda x: x[1], reverse=True)
    return TOP_IDS
