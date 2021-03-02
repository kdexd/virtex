How to evaluate on downstream tasks?
====================================

In our paper, we evaluate our pretrained VirTex models on seven different
downstream tasks. Our codebase supports all of these evaluations. Throughout
this documentation, we consider a specific example of our VirTex pretrained
model being evaluated for ensuring filepath uniformity in the following example
command snippets. Paths can be trivially adjusted for any other VirTex model;
evaluating the baselines (MoCo, ImageNet-supervised, Random Init) require
additional changes in commands, explained in the last sub-section.

As an example, consider a pretraining job for our best performing VirTex model
(``width_ablations/bicaptioning_R_50_L1_H2048.yaml``). The serialization
directory might look something like this:

.. code-block:: text

    /tmp/bicaptioning_R_50_L1_H2048
        pretrain_config.yaml
        log-rank0.txt    # stdout/stderr per GPU process
        log-rank1.txt
        ...
        log-rank7.txt
        checkpoint_2000.pth
        checkpoint_4000.pth
        ...
        checkpoint_498000.pth
        checkpoint_500000.pth    # serialized checkpoints
        train_captioning_forward/
            events.out.* ...    # tensorboard logs
        ...

We evaluate all checkpoints on **PASCAL VOC 2007 Linear Classification**, and
then evaluate the best checkpoint (here, it was iteration 500000) on all other
downstream tasks.


PASCAL VOC 2007 Linear Classification
-------------------------------------

Evaluate a single VirTex pretrained checkpoint on VOC 2007 ``trainval`` split:

.. code-block:: shell

    python scripts/clf_voc07.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --down-config configs/downstream/voc07_clf.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --weight-init virtex \
        --num-gpus-per-machine 1 \
        --cpu-workers 4 \
        --serialization-dir /tmp/bicaptioning_R_50_L1_H2048

To evaluate recent 100 checkpoints in the sub-directory, this command can be
looped over as follows:

.. code-block:: shell

    for ((iter = 300000; iter <= 500000; iter+=2000)); do
        # add command with `checkpoint_$iter.pth`        
    done

This script write metric to tensorboard logs in the same pretraining directory,
all VOC07 mAP curves appear together with pretraining loss curves.

-------------------------------------------------------------------------------

ImageNet Linear Classification
------------------------------

We train a linear classifier on 2048-dimensional global average pooled features
extracted from a frozen visual backbone. Evaluate a checkpoint (for example,
iteration 500000) on this task as:

.. code-block:: shell

    python scripts/clf_linear.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --down-config configs/downstream/imagenet_clf.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --weight-init virtex \
        --num-gpus-per-machine 8 \
        --cpu-workers 4 \
        --serialization-dir /tmp/bicaptioning_R_50_L1_H2048/imagenet_500000 \
        --checkpoint-every 5005  # 1 epoch of ImageNet

-------------------------------------------------------------------------------

Instance Segmentation (and Object Detection) on COCO
----------------------------------------------------

Train a Mask R-CNN with FPN backbone for COCO Instance Segmentation (and Object
Detection, because it also has a box head) by initializing the backbone from
VirTex pretrained weights:

.. code-block:: shell

    python scripts/eval_detectron2.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --d2-config configs/detectron2/coco_segm_default_init_2x.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --weight-init virtex \
        --num-gpus-per-machine 8 \
        --cpu-workers 2 \
        --serialization-dir /tmp/bicaptioning_R_50_L1_H2048/coco_segm_500000 \
        --checkpoint-every 5000

.. note::

    1. This script periodically serializes checkpoints but skips validation
       step during training for saving time; to evaluate a serialized checkpoint
       and write results to tensorboard, provide it as ``--checkpoint-path`` and
       additional flags ``--resume --eval-only``.

    2. Note that ``--d2-config`` here is in Detectron2 format, and not our
       package :class:`~virtex.config.Config`.

    These points are applicable for all tasks described below.

-------------------------------------------------------------------------------

Instance Segmentation on LVIS
-----------------------------

Train a Mask R-CNN with FPN backbone for LVIS Instance Segmentation by
initializing the backbone from VirTex pretrained weights:

.. code-block:: shell

    python scripts/eval_detectron2.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --d2-config configs/detectron2/lvis_segm_default_init_2x.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --weight-init virtex \
        --num-gpus-per-machine 8 \
        --cpu-workers 2 \
        --serialization-dir /tmp/bicaptioning_R_50_L1_H2048/lvis_segm_500000 \
        --checkpoint-every 5000

-------------------------------------------------------------------------------

Object Detection on PASCAL VOC 2007+12
--------------------------------------

Train a Faster R-CNN with C4 backbone for PASCAL VOC 2007+12 Object Detection
by initializing the backbone from VirTex pretrained weights:

.. code-block:: shell

    python scripts/eval_detectron2.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --d2-config configs/detectron2/voc_det_default_init_24k.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --weight-init virtex \
        --num-gpus-per-machine 8 \
        --cpu-workers 2 \
        --serialization-dir /tmp/bicaptioning_R_50_L1_H2048/voc_det_500000 \
        --checkpoint-every 2500

-------------------------------------------------------------------------------

iNaturalist 2018 Fine-Grained Classification
--------------------------------------------

Fine-tune the VirTex pretrained visual backbone end-to-end on iNaturalist 2018
dataset:

.. code-block:: shell

    python scripts/clf_linear.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --down-config configs/downstream/inaturalist_clf.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --weight-init virtex \
        --num-gpus-per-machine 8 \
        --cpu-workers 4 \
        --serialization-dir /tmp/bicaptioning_R_50_L1_H2048/inaturalist_500000 \
        --checkpoint-every 1710  # 1 epoch of iNaturalist

-------------------------------------------------------------------------------

Image Captioning on COCO Captions val2017
-----------------------------------------

Evaluate a pretrained VirTex model on image captioning for COCO Captions val2017
split (reporting CIDEr and SPICE metics):

.. code-block:: shell

    python scripts/eval_captioning.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --calc-metrics \
        --num-gpus-per-machine 1 \
        --cpu-workers 4

-------------------------------------------------------------------------------

Running Image Captioning Inference on Arbitrary Images
------------------------------------------------------

The above script can be used for generating captions for any images in a directory.
Replace certain commands as follows:

.. code-block:: shell

    python scripts/eval_captioning.py \
        --config /tmp/bicaptioning_R_50_L1_H2048/pretrain_config.yaml \
        --checkpoint-path /tmp/bicaptioning_R_50_L1_H2048/checkpoint_500000.pth \
        --data-root /path/to/images_dir \
        --output /path/to/save/predictions.json \
        --num-gpus-per-machine 1 \
        --cpu-workers 4

This script will save predictions in JSON format. Since our goal is to not
improve image captioning, these models may not generate the best captions.
