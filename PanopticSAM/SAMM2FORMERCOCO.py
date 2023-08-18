#@title Utils { form-width: "25%" }
import matplotlib.pyplot as plt
from typing import Optional, List
from torch import Tensor
import torch
import torchvision
import PIL
from torchvision.ops.boxes import box_area
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
    
import torch.nn.functional as F
import torch.nn as nn

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
    


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#@title Hungarian matcher { form-width: "25%" }
from scipy.optimize import linear_sum_assignment
from detectron2.projects.point_rend.point_features import point_sample

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# batch_dice_loss_jit = torch.jit.script(
#     batch_dice_loss
# )  # type: torch.jit.ScriptModule
batch_dice_loss_jit = batch_dice_loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


# batch_sigmoid_ce_loss_jit = torch.jit.script(
#     batch_sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule
batch_sigmoid_ce_loss_jit = batch_sigmoid_ce_loss

from torch.cuda.amp import autocast

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
        
def build_matcher(args):
    return HungarianMatcher(num_points = args.num_points,
    cost_class = args.cost_class,
    cost_mask = args.cost_mask,
    cost_dice = args.cost_dice)

#@title Model - Positional encoding { form-width: "25%" }
import math
from torch import nn

#@title Model - Segmentation { form-width: "25%" }
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict

import torch.nn.functional as F
from torch import Tensor
from PIL import Image

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass





class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """

        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks = outputs["pred_logits"], outputs["pred_masks"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, size, target_size in zip(
            out_logits, raw_masks, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)

            h, w = cur_masks.shape[-2:]

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img


            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )

                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            def clamp(n, minn, maxn):
                return max(min(maxn, n), minn)

            segments_info = []
            for i, a in enumerate(area):
                #cat = clamp(cur_classes[i].item(), 0, 33)
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
#@title Model - Transformer { form-width: "25%" }

"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


# dice_loss_jit = torch.jit.script(
#     dice_loss
# )  # type: torch.jit.ScriptModule
dice_loss_jit = dice_loss

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


# sigmoid_ce_loss_jit = torch.jit.script(
#     sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule
sigmoid_ce_loss_jit = sigmoid_ce_loss

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio=3, importance_sample_ratio=0.75):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = outputs['pred_logits']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        results = [{'scores': s, 'labels': l} for s, l in zip(scores, labels)]

        return results
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
    
    
    
    
from samlibrary import build_sam
from sam_lora import LoRA_Sam
from samlibrary.mask_decoder_adapter import BaseMaskDecoderAdapter, SemMaskDecoderAdapter, PanMaskDecoderAdapter
from samlibrary.prompt_encoder_adapter import BasePromptEncoderAdapter
sam = build_sam.build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth") 
from sam_lora import LoRA_Sam
lora_sam = LoRA_Sam(sam,r=4)
device = "cuda"
class PanopticSAMModel(nn.Module):
    def __init__(self,sam, num_classes, num_queries):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = BasePromptEncoderAdapter(sam,fix=False)
        self.mask_decoder = PanMaskDecoderAdapter(sam,fix=False, num_queries=num_queries)
        # self.all_queries_to_single = nn.Conv2d(num_queries,1,1)
        self.class_conv = nn.Sequential(
            
                nn.Conv2d(100,100,3,stride = 2),##[bs,100,128,128]
                LayerNorm(100),
                nn.GELU(),
                nn.Conv2d(100,100,3,stride = 2),##[bs,100,63,63],
                LayerNorm(100),
                nn.GELU()
            
        )
        self.class_linear = nn.Linear(63*63,num_classes+1)
        self.num_decoder_steps = 3
        
    def forward(self,x):
        if isinstance(x, (list, torch.Tensor)):
            x = nested_tensor_from_tensor_list(x)
        # src, mask = features[-1].decompose()
        x,_ = x.decompose()
        img_embeddings = self.image_encoder(x)##[bs,256,64,64]
        
        
        ##First Decoding Step
        sparse_embeds,dense_embeds = self.prompt_encoder(points = None,boxes = None,masks = None)
        low_res_masks = self.mask_decoder(
            image_embeddings = img_embeddings,
            prompt_adapter= self.prompt_encoder,
            sparse_embeddings=sparse_embeds,
            dense_embeddings=dense_embeds,
            multimask_output=True
        )##[bs,100,256,256]

        class_conv_out = self.class_conv(low_res_masks)##[bs,100,63,63]
        class_conv_out = class_conv_out.flatten(2)##[bs,100,63*63]
        class_preds = self.class_linear(class_conv_out) ##[bs,100,num_classes + 1]
        out = {"pred_logits": class_preds,"pred_masks": low_res_masks}
        
        return out



#@title Dataset - Transform { form-width: "25%" }
"""
Transforms and data augmentation for both image + bbox.
"""
import random
from collections import defaultdict, deque
import PIL
import torchvision.transforms as T
import torchvision.transforms.functional as tvF

def crop(image, target, region):
    cropped_image = tvF.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = tvF.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        # if isinstance(size, (list, tuple)):
        #     return size[::-1]
        # else:
        #     return get_size_with_aspect_ratio(image_size, size, max_size)
        return (size,size)

    size = get_size(image.size, size, max_size)
    rescaled_image = tvF.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = tvF.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return tvF.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = tvF.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

def make_coco_transforms(image_set):
    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return Compose([
            RandomResize([1024], max_size=None),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            RandomResize([1024], max_size=None),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
#@title Dataset - COCO panoptic { form-width: "25%" }
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from panopticapi.utils import rgb2id

city2int = {
    "aachen": 0,
    "bremen": 1,
    "darmstadt": 2,
    "erfurt": 3,
    "hanover": 4,
    "krefeld": 5,
    "strasbourg": 6,
    "tubingen": 7,
    "weimar": 8,
    "bochum": 9,
    "cologne": 10,
    "dusseldorf": 11,
    "hamburg": 12,
    "jena": 13,
    "monchengladbach": 14,
    "stuttgart": 15,
    "ulm": 16,
    "zurich": 17,
    "frankfurt": 18,
    "lindau": 19,
    "munster":20,
    "berlin": 21,
    "bielefeld": 22,
    "bonn": 23,
    "leverkusen": 24,
    "mainz": 25,
    "munich": 26
}

int2city = {v: k for k, v in city2int.items()}

def imgid2int(id):
    city, f, s = id.split('_')
    return int(int(s) + int(f)*1e6 + city2int[city]*1e12)

def int2imgid(num):
    cityn = num // int(1e12)
    f = (num - int(cityn * 1e12)) // int(1e6)
    s = num % int(1e6)
    return int2city[cityn]+'_'+str(f).zfill(6)+'_'+str(s).zfill(6)

class COCOPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / (ann_info['file_name'].replace("png","jpg"))
        ann_path = Path(self.ann_folder) / ann_info['file_name']

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco['images'])

    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width

def build_coco_panoptic(image_set, args):
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    assert img_folder_root.exists(), f'provided path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided path {ann_folder_root} does not exist'
    
    ann_file = {
        "train": "../COCO/coco_ann2017/annotations/panoptic_train2017.json",
        "val": "../COCO/coco_ann2017/annotations/panoptic_val2017.json"
    }

    img_folder_path = {
        "train": "../COCO/coco_train2017/train2017",
        "val": "../COCO/coco_val2017/val2017"
    }

    ann_folder = {
        "train": "../COCO/panoptic_train2017",
        "val": "../COCO/panoptic_val2017",
    }

    dataset = COCOPanoptic(img_folder_path[image_set], ann_folder[image_set], ann_file[image_set],
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset

import torch.utils.data
import torchvision

def build_dataset(image_set, args):
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
#@title Evaluate { form-width: "25%" }
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass
import wandb
import time
class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []

    def update(self, predictions):
        for p in predictions:
            with open(os.path.join(self.output_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))
        self.predictions += predictions

    def synchronize_between_processes(self):
        return

    def summarize(self):
        json_data = {"annotations": self.predictions}
        predictions_json = os.path.join(self.output_dir, "predictions.json")
        with open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))
        return pq_compute(self.gt_json, predictions_json, gt_folder=self.gt_folder, pred_folder=self.output_dir)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        return


    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)
        
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def log_class_chart(results):
    id2name  = {
    0: 'unlabeled',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
    92: 'banner',
    93: 'blanket',
    95: 'bridge',
    100: 'cardboard',
    107: 'counter',
    109: 'curtain',
    112: 'door-stuff',
    118: 'floor-wood',
    119: 'flower',
    122: 'fruit',
    125: 'gravel',
    128: 'house',
    130: 'light',
    133: 'mirror-stuff',
    138: 'net',
    141: 'pillow',
    144: 'platform',
    145: 'playingfield',
    147: 'railroad',
    148: 'river',
    149: 'road',
    151: 'roof',
    154: 'sand',
    155: 'sea',
    156: 'shelf',
    159: 'snow',
    161: 'stairs',
    166: 'tent',
    168: 'towel',
    171: 'wall-brick',
    175: 'wall-stone',
    176: 'wall-tile',
    177: 'wall-wood',
    178: 'water-other',
    180: 'window-blind',
    181: 'window-other',
    184: 'tree-merged',
    185: 'fence-merged',
    186: 'ceiling-merged',
    187: 'sky-other-merged',
    188: 'cabinet-merged',
    189: 'table-merged',
    190: 'floor-other-merged',
    191: 'pavement-merged',
    192: 'mountain-merged',
    193: 'grass-merged',
    194: 'dirt-merged',
    195: 'paper-merged',
    196: 'food-other-merged',
    197: 'building-other-merged',
    198: 'rock-merged',
    199: 'wall-other-merged',
    200: 'rug-merged'
}

    cl = results['per_class']

    plt.figure(figsize=(10, 20), dpi=80)

    for k in cl.keys():
        c = cl[k]
        plt.barh(id2name[k]+'_pq', c['pq'], color = 'violet')
        plt.barh(id2name[k]+'_sq', c['sq'], color = 'red')
        plt.barh(id2name[k]+'_rq', c['rq'], color = 'blue')

    plt.savefig('chart.png')
    plt.show()
    wandb.log({"chart": wandb.Image(Image.open('chart.png'))})

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
        
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = (target["image_id"].item())
                file_name = f"{image_id}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    log_class_chart(panoptic_res)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats

#@title Train loop { form-width: "25%" }
"""
Train and eval functions used in main.py
"""
from torch.utils.data import DataLoader, DistributedSampler
import math
import sys
import time 
import datetime
from typing import Iterable

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10 
    # ctx = torch.cuda.amp.autocast()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # with ctx:
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ENABLE_WANDB:
            wandb.log(loss_dict_reduced)
            wandb.log({'loss':loss_value})
            
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train():

    # !mkdir out_dir/panoptic_eval -p
    model = PanopticSAMModel(lora_sam.sam,201,100)
    model.to(device)
    matcher = build_matcher(args)
    losses = ['labels', 'masks']
    criterion = SetCriterion(201,matcher,args.weight_dict,args.eos_coef,losses,args.num_points)
    criterion.to(device)
    is_thing_map = {i: i <= 80 for i in range(201)}
    postprocessors = {}
    postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    postprocessors['segm'] = PostProcessSegm()

    if ENABLE_WANDB:
        wandb.watch(model)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if ENABLE_WANDB and used_artifact is not None:
        artifact = wandb.use_artifact(used_artifact)
        artifact_dir = artifact.download()
        checkpoint = torch.load(artifact_dir+"/checkpoint.pth")

        model.load_state_dict(checkpoint['model'])
        if run_id is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=collate_fn, num_workers=1)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=collate_fn, num_workers=1)


    output_dir = Path(args.output_dir)

    if args.eval:
        test_stats = evaluate(model, criterion, postprocessors,
                                                data_loader_val, device, args.output_dir)
        print(test_stats)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch+1,args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_path = output_dir / 'checkpoint.pth'
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            if ENABLE_WANDB:
                artifact = wandb.Artifact(wandb_experiment_name, type='model')
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)

        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        if ENABLE_WANDB:
            wandb.log(test_stats)

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
#@title Init { form-width: "25%" }
import random
import numpy as np
import torch

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)     # missing some deterministic impl

device = torch.device("cuda:0")

class Args:
    pass
args = Args()
# Hungarian matcher
args.cost_class = 2
args.cost_mask = 5
args.cost_dice = 5
# DETR
args.num_queries = 100
args.aux_loss = False
args.masks = True
args.weight_dict = {'loss_ce': 2, 'loss_mask': 5, "loss_dice":5}
args.eos_coef = 0.1
# Dataset
args.dataset_file = "coco_panoptic" # cityscape
args.coco_path = ""
args.coco_panoptic_path = ""
# Training
args.lr = 1e-4
args.weight_decay = 1e-4
args.batch_size = 1
args.epochs = 10
args.lr_drop = 6
args.clip_max_norm = 0.1

args.num_points = 12544
args.output_dir = "out_dir"
args.eval = False

#@title Transfer learning policies { form-width: "25%" }








#@title WandB { form-width: "25%" }

ENABLE_WANDB = True                                      # set if you plan to log on wandb
# used_artifact = 'circles/detr/second:latest'            # if set not train from scratch (detre pretrained on COCO)
used_artifact = None
wandb_experiment_name = "SAM_b_v1.0"      # set if starting a new run
run_id = None                                        # set to None if starting a new run

if ENABLE_WANDB:
    # !pip install wandb > /dev/null
    # !wandb login
    import wandb
    
    if run_id is not None: 
        wandb.init(project='SAMCOCO', id=run_id, resume='allow')
    else:
        wandb.init(project='SAMCOCO', name=wandb_experiment_name)
    
    wandb.config.cost_class = args.cost_class
    wandb.config.cost_mask = args.cost_mask
    wandb.config.cost_dice = args.cost_dice

    wandb.config.num_queries = args.num_queries
    wandb.config.aux_loss = args.aux_loss
    wandb.config.masks = args.masks
    wandb.config.weight_dict = args.weight_dict
    wandb.config.eos_coef = args.eos_coef
    wandb.config.lr = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.batch_size = args.batch_size
    wandb.config.epochs = args.epochs
    wandb.config.lr_drop = args.lr_drop
    wandb.config.clip_max_norm = args.clip_max_norm
    

train()