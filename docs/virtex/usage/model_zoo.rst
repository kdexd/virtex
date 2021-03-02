VirTex Model Zoo
================

We provide a collection of pretrained model weights and corresponding config
names in this model zoo. Tables contain partial paths to config files for each
model, download link for pretrained weights and for reference -- VOC07 mAP and
ImageNet top-1 accuracy.

The simplest way to download and use a *full* pretrained model (including both,
the visual backbone and the textual head) is through :doc:`../model_zoo` API as
follows. This code snippet works from anywhere, and does not require to be
executed from project root.

.. code-block:: python

    # Get our full best performing VirTex model:
    import virtex.model_zoo as mz
    model = mz.get("width_ablations/bicaptioning_R_50_L1_H2048.yaml", pretrained=True)

    # Optionally extract the torchvision-like visual backbone (with ``avgpool``
    # and ``fc`` layers replaced with ``nn.Identity`` module).
    cnn = model.visual.cnn

Alternatively, weights can be manually downloaded from links below, and this
can be executed from the project root:

.. code-block:: python

    from virtex.config import Config
    from virtex.factories import PretrainingModelFactory
    from virtex.utils.checkpointing import CheckpointManager

    # Get the best performing VirTex model:
    _C = Config("configs/width_ablations/bicaptioning_R_50_L1_H2048.yaml")
    model = PretrainingModelFactory.from_config(_C)

    CheckpointManager(model=model).load("/path/to/downloaded/weights.pth")

    # Optionally extract the torchvision-like visual backbone (with ``avgpool``
    # and ``fc`` layers replaced with ``nn.Identity`` module).
    cnn = model.visual.cnn


The pretrained ResNet-50 visual backbone of our best performing model
(``width_ablations/bicaptioning_R_50_L1_H2048.yaml``) can be loaded in a single
line, *without following any installation steps* (only requires PyTorch v1.5):

.. code-block:: python

    import torch

    model = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True)

    # This is a torchvision-like resnet50 model, with ``avgpool`` and ``fc``
    # layers replaced with ``nn.Identity`` module.
    image_batch = torch.randn(1, 3, 224, 224)  # batch tensor of one image.
    features_batch = model(image_batch)  # shape: (1, 2048, 7, 7)

-------------------------------------------------------------------------------

Pretraining Task Ablations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-zlqz{background-color:#d5d5d5;border-color:inherit;font-weight:bold;text-align:center;vertical-align:center}
    .tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
    .tg .tg-c3ow a{color: darkgreen; text-decoration: none; border-bottom: 1px dashed green;text-underline-position: under;
    .tg .tg-c3ow a:hover{font-weight: 700;border-bottom: 1px solid green;}
    .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
    @media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;}}</style>
    <div class="tg-wrap"><table class="tg">
    <tbody>
    <tr>
        <td class="tg-zlqz">Model Config Name</td>
        <td class="tg-zlqz">VOC07<br>mAP</td>
        <td class="tg-zlqz">ImageNet<br>Top-1 Acc.</td>
        <td class="tg-zlqz">Model URL</td>
    </tr>
    <tr>
        <td class="tg-0pky">task_ablations/bicaptioning_R_50_L1_H2048.yaml</td>
        <td class="tg-c3ow">88.7</td>
        <td class="tg-c3ow">53.8</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/zu8zxtxrron29icd76owgjzojmfcgdk3.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">task_ablations/captioning_R_50_L1_H2048.yaml</td>
        <td class="tg-c3ow">88.6</td>
        <td class="tg-c3ow">50.8</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/1q9qh1cj2u4r5laj7mefd2mlzwthnga7.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">task_ablations/token_classification_R_50.yaml</td>
        <td class="tg-c3ow">88.8</td>
        <td class="tg-c3ow">48.6</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/idvoxjl60pzpcllkbvadqgvwazil2mis.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">task_ablations/multilabel_classification_R_50.yaml</td>
        <td class="tg-c3ow">86.2</td>
        <td class="tg-c3ow">46.2</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/yvlflmo0klqy3m71p6ug06c6aeg282hy.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">task_ablations/masked_lm_R_50_L1_H2048.yaml</td>
        <td class="tg-c3ow">86.4</td>
        <td class="tg-c3ow">46.7</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/x3eij00eslse9j35t9j9ijyj8zkbkizh.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    </tbody>
    </table></div>


Width Ablations
^^^^^^^^^^^^^^^

.. raw:: html

    <div class="tg-wrap"><table class="tg">
    <tbody>
    <tr>
        <td class="tg-zlqz">Model Config Name</td>
        <td class="tg-zlqz">VOC07<br>mAP</td>
        <td class="tg-zlqz">ImageNet<br>Top-1 Acc.</td>
        <td class="tg-zlqz">Model URL</td>
    </tr>
    <tr>
        <td class="tg-0pky">width_ablations/bicaptioning_R_50_L1_H512.yaml</td>
        <td class="tg-c3ow">88.4</td>
        <td class="tg-c3ow">51.8</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/wtk18v0vffws48u5yrj2qjt94wje1pit.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky"><span style="font-weight:400;font-style:normal">width_ablations/bicaptioning_R_50_L1_H768.yaml</span></td>
        <td class="tg-c3ow">88.3</td>
        <td class="tg-c3ow">52.3</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/e94n0iexdvksi252bn7sm2vqjnyt9okf.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky"><span style="font-weight:400;font-style:normal">width_ablations/bicaptioning_R_50_L1_H1024.yaml</span></td>
        <td class="tg-c3ow">88.3</td>
        <td class="tg-c3ow">53.2</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/1so9cu9y06gy27rqbzwvek4aakfd8opf.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky"><span style="font-weight:400;font-style:normal">width_ablations/bicaptioning_R_50_L1_H2048.yaml</span></td>
        <td class="tg-c3ow">88.7</td>
        <td class="tg-c3ow">53.8</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/zu8zxtxrron29icd76owgjzojmfcgdk3.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    </tbody>
    </table></div>


Depth Ablations
^^^^^^^^^^^^^^^

.. raw:: html

    <div class="tg-wrap"><table class="tg">
    <tbody>
    <tr>
        <td class="tg-zlqz">Model Config Name</td>
        <td class="tg-zlqz">VOC07<br>mAP</td>
        <td class="tg-zlqz">ImageNet<br>Top-1 Acc.</td>
        <td class="tg-zlqz">Model URL</td>
    </tr>
    <tr>
        <td class="tg-0pky">depth_ablations/bicaptioning_R_50_L1_H1024.yaml</td>
        <td class="tg-c3ow">88.3</td>
        <td class="tg-c3ow">53.2</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/1so9cu9y06gy27rqbzwvek4aakfd8opf.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">depth_ablations/bicaptioning_R_50_L2_H1024.yaml</td>
        <td class="tg-c3ow">88.8</td>
        <td class="tg-c3ow">53.8</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/9e88f6l13a9r8wq5bbe8qnoh9zenanq3.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky"><span style="font-weight:400;font-style:normal">depth_ablations/bicaptioning_R_50_L3_H1024.yaml</span></td>
        <td class="tg-c3ow">88.7</td>
        <td class="tg-c3ow">53.9</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/4cv8052xiq91h7lyx52cp2a6m7m9qkgo.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky"><span style="font-weight:400;font-style:normal">depth_ablations/bicaptioning_R_50_L4_H1024.yaml</span></td>
        <td class="tg-c3ow">88.7</td>
        <td class="tg-c3ow">53.9</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/bk5w4471mgvwa5mv6e4c7htgsafzmfm0.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    </tbody>
    </table></div>


Backbone Ablations
^^^^^^^^^^^^^^^^^^

.. raw:: html

    <div class="tg-wrap"><table class="tg">
    <tbody>
    <tr>
        <td class="tg-zlqz">Model Config Name</td>
        <td class="tg-zlqz">VOC07<br>mAP</td>
        <td class="tg-zlqz">ImageNet<br>Top-1 Acc.</td>
        <td class="tg-zlqz">Model URL</td>
    </tr>
    <tr>
        <td class="tg-0pky">backbone_ablations/bicaptioning_R_50_L1_H1024.yaml</td>
        <td class="tg-c3ow">88.3</td>
        <td class="tg-c3ow">53.2</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/1so9cu9y06gy27rqbzwvek4aakfd8opf.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">backbone_ablations/bicaptioning_R_50W2X_L1_H1024.yaml</td>
        <td class="tg-c3ow">88.5</td>
        <td class="tg-c3ow">52.9</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/19vcaf1488945836kir9ebm5itgtugaw.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    <tr>
        <td class="tg-0pky">backbone_ablations/bicaptioning_R_101_L1_H1024.yaml</td>
        <td class="tg-c3ow">88.7</td>
        <td class="tg-c3ow">52.1</td>
        <td class="tg-c3ow"><a href="https://umich.box.com/shared/static/nptbh4jsj0c0kjsnc2hw754fkikpgx9v.pth" target="_blank" rel="noopener noreferrer">model</a></td>
    </tr>
    </tbody>
    </table></div>
