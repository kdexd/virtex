CHANGELOG
=========

This CHANGELOG file records changes between different arXiv versions of our paper, and the version of this codebase which should be used to reproduce the results in the corresponding arXiv version. View changes between code versions on the [Releases page](https://github.com/kdexd/virtex/releases).

ArXiv v1 -> v2
==============

**Code version:** `v1.2`.

Fix image captioning results with a modified beam search implementation. _Rest of the downstream task results and pre-trained models are unchanged._


ArXiv v1 -> v2
==============

**Code version:** `v1.0` or `v1.1`.

[ArXiv v1](https://arxiv.org/abs/2006.06666v1) was our ECCV 2020 submission (reject). [ArXiv v2](https://arxiv.org/abs/2006.06666v2) is our CVPR 2021 submission (accept). The repository snapshots for these two versions are tagged at [`v0.9`](https://github.com/kdexd/virtex/releases/tag/v0.9) and [`v1.0`](https://github.com/kdexd/virtex/releases/tag/v1.0).

While the core motivation and approach is the same, we have made some minor changes in our experiments and evaluation setup. These slightly improve model performances across the board (within decimals). New models are available in [`v1.0` model zoo](http://kdexd.github.io/virtex/virtex/usage/model_zoo.html), however links to old models in `v0.9` will be active till June 30, 2021. We encourage you to use the new models!

We have updated the experiment config files for all changes described below.

Experiment Changes
------------------

### New Feature:

Add a new pretraining task for BERT-style _Masked Language Modeling_. Pre-trained model released in Model Zoo.

### Pre-training:

- The only change during pre-training is that we do not apply weight decay to LayerNorm and biases in input embedding and transformer layers. We apply weight decay to the biases in output linear layer (before softmax).

- Other factors that could affect results:
  - Use official [albumentations.ColorJitter transform](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter) that mimics torchvision ColorJitter transform. Earlier I implemented [my own ColorJitter](https://github.com/kdexd/virtex/blob/c19e7fc9b98e98af82286ed1537b6f588eaeac44/virtex/data/transforms.py#L156) because albumentations didn't have one.
  - Use PyTorch Native AMP (Automatic Mixed Precision) instead of NVIDIA Apex.

### Downstream Evaluations:

1. **PASCAL VOC 2007 Linear Classification:** [[diff]](https://github.com/kdexd/virtex/compare/57889ca9829f27b932e92b9e6b51f50f20f2d546..7645cc0d1e3e49f00e347e9873fd020faa2ec62e#diff-b4405dd4879a48ef1e5b1e2801035909584a5f1f32f63d5e793fb50dee077b97)
   - Instead of training linear SVMs on 8192-dimensional average pooled features from ResNet-50 (7x7x2048 —> 2x2x2048), like [(Misra et al. 2019)](https://arxiv.org/abs/1905.01235), we directly train SVMs on 2048-dimensional global average pooled features, following recent works like [SwAV (Caron et al. 2020)](https://arxiv.org/abs/2006.09882).
   - We change the pre-processing: resize shortest edge to 256 pixels, and take center crop of 224 pixels.
   - These improve VOC mAP by 1-2 points everywhere, and makes SVM training faster. Since we select best checkpoint based on this metric, all results on other downstream tasks also change in `ArXiv v2` (But the trends remain same.)

2. **ImageNet Linear Evaluation:** [[diff]](https://github.com/kdexd/virtex/compare/57889ca9829f27b932e92b9e6b51f50f20f2d546..7645cc0d1e3e49f00e347e9873fd020faa2ec62e#diff-d3dea1e7bf97d0cfca4b59a47c0a9bb81e78b8827654fe0258df9ce2c3f5f41c)
   - Changed random resized crop scale from (20-100%) to (8-100%) for consistency with evaluations in SSL works like MoCo and SwAV.
   - Use cosine LR decay instead of step decay, following SwAV. Improves accuracy by up to 1%.

3. **iNaturalist Fine-tuning:** [[diff]](https://github.com/kdexd/virtex/compare/57889ca9829f27b932e92b9e6b51f50f20f2d546..7645cc0d1e3e49f00e347e9873fd020faa2ec62e#diff-09096da78cfcde3a604ce22d80313f0800225d928cce5ef7334b89a382adfe4d)
   - This evaluation is left unchanged across ArXiv versions, but we fixd a typo in image pre-processing step, present in publicly released config.

4. **Detectron2 tasks (COCO and LVIS Instance Segmentation, VOC Detection):**
   - Heavily simplified the script. Updated Detectron2 uses a more memory-efficient SyncBatchNorm and supports AMP.

