import hashlib

import torch
import torch.nn.functional as F
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None



class dice_loss:
    """ Dice Loss class function:
    """

    def loss(self,y_true,y_pred,loss_mult=None):
        smooth = 1.
        iflat = y_pred.view(-1)
        tflat = y_true.view(-1)

        intersection = (iflat * tflat).sum()
        dice = 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

        if loss_mult is not None:
            dice *= loss_mult

        return dice

    def np_loss(self,y_true,y_pred,loss_mult=None):
        
        return self.loss(y_true,y_pred,loss_mult).item()


class grad_loss:
    """Grad Loss function:

    Grad loss using the absolute loss (e.g. with the grid too).
    Adapted from: https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self,params,penalty='l2'):

        self.penalty=penalty
        self.ndims = params.dataset.ndims

    def loss(self,_,y_pred,loss_mult=None):
        
        '''
        Using Pytorch grid format: e.g. between -1 and 1
        '''

        size = np.shape(y_pred)[2:]
        device = y_pred.device
        dtype = y_pred.dtype
        vectors = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        flow_feild = y_pred + grid.unsqueeze(0)

        dy = torch.abs(flow_feild[:, :, 1:, : ] - flow_feild[:, :, :-1, :]) 
        dx = torch.abs(flow_feild[:, :, :, 1:] - flow_feild[:, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy) 
        grad = d / 2.0

        if loss_mult is not None:
            grad *= loss_mult

        return grad

    def np_loss(self,_,y_pred,loss_mult=None):

        return self.loss(_,y_pred,loss_mult).item()


def _align_prediction_and_target(y_true, y_pred):
    if y_true.ndim == y_pred.ndim - 1:
        y_true = y_true.unsqueeze(1)
    if y_pred.ndim == y_true.ndim - 1:
        y_pred = y_pred.unsqueeze(1)
    if y_true.shape[2:] != y_pred.shape[2:]:
        y_true = F.interpolate(y_true.float(), size=y_pred.shape[2:], mode="nearest")
    return y_true.float(), y_pred.float()


class flow_smoothness_loss:
    """Direct smoothness penalty on displacement-field spatial gradients."""

    def __init__(self, penalty="l2"):
        self.penalty = str(penalty).lower()
        if self.penalty not in {"l1", "l2"}:
            raise ValueError(f"Unsupported flow smoothness penalty '{penalty}'.")

    def loss(self, _, y_pred, loss_mult=None):
        diffs = []
        for dim in range(2, y_pred.ndim):
            forward = y_pred.narrow(dim, 1, y_pred.shape[dim] - 1)
            backward = y_pred.narrow(dim, 0, y_pred.shape[dim] - 1)
            diff = torch.abs(forward - backward)
            if self.penalty == "l2":
                diff = diff * diff
            diffs.append(torch.mean(diff))

        smooth = sum(diffs) / max(len(diffs), 1)
        if loss_mult is not None:
            smooth *= loss_mult
        return smooth

    def np_loss(self, _, y_pred, loss_mult=None):
        return self.loss(_, y_pred, loss_mult).item()


class boundary_distance_loss:
    """
    Distance-transform weighted mask loss.

    The target signed distance map is computed once per batch from the binary
    label. Gradients flow through the soft prediction, not through the distance
    transform itself.
    """

    def __init__(self, max_distance=20.0, min_weight=1.0, eps=1e-6):
        if distance_transform_edt is None:
            raise ImportError("scipy is required for boundary_distance_loss.")
        self.max_distance = float(max_distance)
        self.min_weight = float(min_weight)
        self.eps = float(eps)
        self._distance_cache = {}

    def _cache_key(self, mask):
        digest = hashlib.blake2b(mask.tobytes(), digest_size=16).hexdigest()
        return mask.shape, digest

    def _compute_signed_distance_map(self, mask):
        if not mask.any():
            return np.ones(mask.shape, dtype=np.float32)
        if mask.all():
            return -np.ones(mask.shape, dtype=np.float32)

        key = self._cache_key(mask)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached

        outside = distance_transform_edt(~mask)
        inside = distance_transform_edt(mask)
        signed_distance = outside - inside
        if self.max_distance > 0:
            signed_distance = np.clip(
                signed_distance,
                -self.max_distance,
                self.max_distance,
            )
        normalizer = max(float(np.max(np.abs(signed_distance))), self.eps)
        signed_distance = (signed_distance / normalizer).astype(np.float32)
        self._distance_cache[key] = signed_distance
        return signed_distance

    def _signed_distance_map(self, y_true):
        target_np = y_true.detach().cpu().numpy()
        maps = np.zeros_like(target_np, dtype=np.float32)

        for index in np.ndindex(target_np.shape[:2]):
            mask = target_np[index] > 0.5
            maps[index] = self._compute_signed_distance_map(mask)

        return torch.from_numpy(maps).to(device=y_true.device, dtype=y_true.dtype)

    def loss(self, y_true, y_pred, loss_mult=None):
        y_true, y_pred = _align_prediction_and_target(y_true, y_pred)
        signed_distance = self._signed_distance_map(y_true)
        distance_weight = self.min_weight + torch.abs(signed_distance)
        boundary_loss = torch.mean(torch.abs(y_pred - y_true) * distance_weight)
        if loss_mult is not None:
            boundary_loss *= loss_mult
        return boundary_loss

    def np_loss(self, y_true, y_pred, loss_mult=None):
        return self.loss(y_true, y_pred, loss_mult).item()


class soft_cldice_loss:
    """Soft clDice topology proxy based on differentiable skeletonization."""

    def __init__(self, iterations=10, eps=1e-6):
        self.iterations = int(iterations)
        self.eps = float(eps)

    def _soft_erode(self, img):
        if img.ndim == 4:
            p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
            p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
            return torch.min(p1, p2)
        if img.ndim == 5:
            p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
            p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        raise ValueError(f"Expected 4D or 5D tensor for soft clDice, got {tuple(img.shape)}")

    def _soft_dilate(self, img):
        if img.ndim == 4:
            return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
        if img.ndim == 5:
            return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)
        raise ValueError(f"Expected 4D or 5D tensor for soft clDice, got {tuple(img.shape)}")

    def _soft_open(self, img):
        return self._soft_dilate(self._soft_erode(img))

    def _soft_skeleton(self, img):
        img = torch.clamp(img, 0.0, 1.0)
        opened = self._soft_open(img)
        skeleton = F.relu(img - opened)
        for _ in range(self.iterations):
            img = self._soft_erode(img)
            opened = self._soft_open(img)
            delta = F.relu(img - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    def loss(self, y_true, y_pred, loss_mult=None):
        y_true, y_pred = _align_prediction_and_target(y_true, y_pred)
        y_true = torch.clamp(y_true, 0.0, 1.0)
        y_pred = torch.clamp(y_pred, 0.0, 1.0)

        pred_skeleton = self._soft_skeleton(y_pred)
        target_skeleton = self._soft_skeleton(y_true)
        reduce_dims = tuple(range(1, y_pred.ndim))

        topological_precision = (
            torch.sum(pred_skeleton * y_true, dim=reduce_dims) + self.eps
        ) / (torch.sum(pred_skeleton, dim=reduce_dims) + self.eps)
        topological_sensitivity = (
            torch.sum(target_skeleton * y_pred, dim=reduce_dims) + self.eps
        ) / (torch.sum(target_skeleton, dim=reduce_dims) + self.eps)
        cldice = (
            2.0 * topological_precision * topological_sensitivity + self.eps
        ) / (topological_precision + topological_sensitivity + self.eps)
        loss_value = torch.mean(1.0 - cldice)
        if loss_mult is not None:
            loss_value *= loss_mult
        return loss_value

    def np_loss(self, y_true, y_pred, loss_mult=None):
        return self.loss(y_true, y_pred, loss_mult).item()
        
