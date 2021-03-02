How to setup this codebase?
===========================

.. raw:: html

    <hr>

This codebase requires Python 3.6+ or higher. We recommend using Anaconda or
Miniconda. We walk through installation and data preprocessing here.


Install Dependencies
--------------------

For these steps to install through Anaconda (or Miniconda).

1. Install Anaconda or Miniconda distribution based on Python 3+ from their
   `downloads site <https://conda.io/docs/user-guide/install/download.html>`_.


2. Clone the repository first.

    .. code-block:: shell

        git clone https://www.github.com/kdexd/virtex


3. Create a conda environment and install all the dependencies.

    .. code-block:: shell

        cd virtex
        conda create -n virtex python=3.6
        conda activate virtex
        pip install -r requirements.txt


4. Install this codebase as a package in development version.

    .. code-block:: shell

        python setup.py develop

Now you can ``import virtex`` from anywhere as long as you have this conda
environment activated.

-------------------------------------------------------------------------------


Setup Datasets
--------------

Datasets are assumed to exist in ``./datasets`` directory (relative to the
project root) following the structure specified below. COCO is used for
pretraining, and rest of the datasets (including COCO) are used for downstream
tasks. This structure is compatible when using
`Detectron2 <https://github.com/facebookresearch/detectron2>`_ for downstream
tasks.

COCO
^^^^
.. code-block::

    datasets/coco/
        annotations/
            captions_{train,val}2017.json
            instances_{train,val}2017.json
        train2017/
            # images in train2017 split
        val2017/
            # images in val2017 split

LVIS
^^^^
.. code-block::

    datasets/coco/
        train2017/
        val2017/
    datasets/lvis/
        lvis_v1.0_{train,val}.json

PASCAL VOC
^^^^^^^^^^
.. code-block::

    datasets/VOC2007/
        Annotations/
        ImageSets/
            Main/
                trainval.txt
                test.txt
        JPEGImages/

    datasets/VOC2012/
        # Same as VOC2007 above

ImageNet
^^^^^^^^
.. code-block::

    datasets/imagenet/
        train/
            # One directory per category with images in it
        val/
            # One directory per category with images in it
        ILSVRC2012_devkit_t12.tar.gz

iNaturalist 2018
^^^^^^^^^^^^^^^^
.. code-block::

    datasets/inaturalist/
        train_val2018/
        annotations/
            train2018.json
            val2018.json

-------------------------------------------------------------------------------


Preprocess Data
---------------

1. Build a vocabulary out of COCO Captions ``train2017`` split.

    .. code-block:: shell

        python scripts/preprocess/build_vocabulary.py \
            --captions datasets/coco/annotations/captions_train2017.json \
            --vocab-size 10000 \
            --output-prefix datasets/vocab/coco_10k \
            --do-lower-case


2. Serialize COCO Captions (``train2017`` and ``val2017`` splits) into LMDB
   files. These are faster for data reading during pretraining.

    .. code-block:: shell

        python scripts/preprocess/preprocess_coco.py \
            --data-root datasets/coco \
            --split train \
            --output datasets/coco/serialized_train.lmdb

    .. code-block:: shell

        python scripts/preprocess/preprocess_coco.py \
            --data-root datasets/coco \
            --split val \
            --output datasets/coco/serialized_val.lmdb

That's it! You are all set to use this codebase.
