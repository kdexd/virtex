.. raw:: html

    <h1 style="text-align: center">
    VirTex: Learning Visual Representations from Textual Annotations
    </h1>
    <h4 style="text-align: center">
    Karan Desai and Justin Johnson
    </br>
    <span style="font-size: 14pt; color: #555555">
    University of Michigan
    </span>
    </h4>
    <hr>

    <h4 style="text-align: center">
    Abstract
    </h4>

    <p style="text-align: justify">
    The de-facto approach to many vision tasks is to start from pretrained
    visual representations, typically learned via supervised training on
    ImageNet. Recent methods have explored unsupervised pretraining to scale to
    vast quantities of unlabeled images. In contrast, we aim to learn
    high-quality visual representations from fewer images. To this end we
    revisit supervised pretraining, and seek data-efficient alternatives to
    classification-based pretraining. We propose VirTex -- a pretraining
    approach using semantically dense captions to learn visual representations.
    We train convolutional networks from scratch on COCO Captions, and transfer
    them to downstream recognition tasks including image classification, object
    detection, and instance segmentation. On all tasks, VirTex yields features
    that match or exceed those learned on ImageNet -- supervised or unsupervised
    -- despite using up to ten times fewer images.
    </p>

**Code available at:** `github.com/kdexd/virtex <https://github.com/kdexd/virtex>`_.

.. image:: _static/system_figure.jpg


Get the pretrained ResNet-50 visual backbone from our best performing VirTex
model in one line *without any installation*!

.. code-block:: python

    import torch

    # That's it, this one line only requires PyTorch.
    model = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True)


More details in :doc:`virtex/usage/model_zoo`. Next, dive deeper into our
code with User Guide and API References!


User Guide
----------

.. toctree::
    :maxdepth: 2

    virtex/usage/setup_dependencies
    virtex/usage/model_zoo
    virtex/usage/pretrain
    virtex/usage/downstream


API Reference
-------------

.. toctree::
    :maxdepth: 2

    virtex/config
    virtex/factories
    virtex/data
    virtex/models
    virtex/modules
    virtex/optim
    virtex/utils
    virtex/model_zoo


Citation
--------

If you find this code useful, please consider citing:

.. code-block:: text

    @article{desai2020virtex,
        title={VirTex: Learning Visual Representations from Textual Annotations},
        author={Karan Desai and Justin Johnson},
        journal={ TODO },
        year={2020}
    }


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
