import torch
import functools

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync
    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
else:
    ##https://github.com/HRNet/HRNet-Semantic-Segmentation/issues/222
    ##If you use single GPU (not distributed mode), do not use syncBN
    # BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
    relu_inplace = True