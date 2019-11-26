import argparse
import numpy as np
import os
import pickle
import sys

from loguru import logger
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import torch


# fmt: off
parser = argparse.ArgumentParser(
    description="""Train SVMs on intermediate features of pre-trained
    ResNet-like models for Pascal VOC2007 classification."""
)
parser.add_argument(
    "--features-file", type=str, default=None,
    help="Numpy file containing image features.",
)
parser.add_argument(
    "--targets-file", type=str, default=None,
    help="Numpy file containing image labels.",
)
parser.add_argument(
    "--output-path", type=str, default=None,
    help="Path where to save the trained SVM models.",
)
parser.add_argument(
    "--costs", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0],
    help="List of costs to train SVM on.",
)
parser.add_argument(
    "--random-seed", type=int, default=0,
    help="Random seed for SVM classifier training.",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--cpu-workers", type=int, default=0,
    help="Number of CPU workers per GPU to use for data loading.",
)
parser.add_argument(
    "--gpu-id", type=int, default=0, help="ID of GPU to use (-1 for CPU)."
)

# parser.add_argument_group("Checkpointing and Logging")
# parser.add_argument(
#     "--checkpoint-path", required=True,
#     help="""Path to load checkpoint and run downstream task evaluation. The
#     name of checkpoint file is required to be `checkpoint_*.pth`, where * is
#     iteration number from which the checkpoint was serialized. This script will
#     log evaluation results in the directory of this checkpoint for this
#     particular iteration."""
# )
# fmt: on


# def extract_features(opts):
#     workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
#     np.random.seed(cfg.RNG_SEED)

#     # create the model in test mode only
#     assert opts.data_type in ['train', 'val', 'test'], "Please specify valid type."
#     model = model_builder.ModelBuilder(
#         name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
#         use_cudnn=True, cudnn_exhaustive_search=True, split=opts.data_type
#     )
#     model.build_model()
#     model.create_net()
#     model.start_data_loader()

#     # initialize the model
#     if cfg.TEST.PARAMS_FILE:
#         checkpoints.load_model_from_params_file(
#             model, params_file=cfg.TEST.PARAMS_FILE, checkpoint_dir=None)
#     else:
#         logger.info('=====WARN: No params files specified for testing model!')

#     # resolve blob names from possibly provided regex patterns
#     extract_blobs = []
#     for blob_re in cfg.MODEL.EXTRACT_BLOBS:
#         compiled_blob_re = re.compile(r"^{}$".format(blob_re))

#         for bl in workspace.Blobs():
#             # remove device id from blob name
#             bl = bl.split("/")[-1]
#             if compiled_blob_re.match(bl):
#                 extract_blobs.append(bl)

#     # keep unique blob names.
#     extract_blobs = list(set(extract_blobs))
#     logger.info("Extracting blobs: {}".format(", ".join(extract_blobs)))

#     # initialize the dictionary to store features and targets
#     img_features, img_targets = {}, {}
#     for bl in extract_blobs:
#         img_features[bl], img_targets[bl] = {}, {}

#     # we keep track of data indices seen so far. This ensures we extract feature
#     # for each image only once.
#     indices_list = []
#     total_test_iters = helpers.get_num_test_iter(model.input_db)
#     logger.info('Test epoch iters: {}'.format(total_test_iters))
#     # when we extract features, we run 4 epochs to make sure we capture all the
#     # data points. This is needed because we use the multi-processing dataloader
#     # which shuffles the data. In very low-shot setting, making multiple passes
#     # over the entire data becomes crucial.
#     extraction_iters = int(total_test_iters * 4)
#     for test_iter in range(0, extraction_iters):
#         workspace.RunNet(model.net.Proto().name)
#         if test_iter == 0:
#             helpers.print_net(model)
#         if test_iter % 100 == 0:
#             logger.info('at: [{}/{}]'.format(test_iter, extraction_iters))
#         for device in range(cfg.NUM_DEVICES):
#             indices = workspace.FetchBlob('gpu_{}/db_indices'.format(device))
#             labels = workspace.FetchBlob('gpu_{}/labels'.format(device))
#             num_images = indices.shape[0]
#             indices_list.extend(list(indices))
#             for bl in extract_blobs:
#                 features = workspace.FetchBlob('gpu_{}/{}'.format(device, bl))
#                 for idx in range(num_images):
#                     index = indices[idx]
#                     if not (index in img_features[bl]):
#                         img_targets[bl][index] = labels[idx].reshape(-1)
#                         img_features[bl][index] = features[idx]

#     for bl in extract_blobs:
#         img_features[bl] = dict(sorted(img_features[bl].items()))
#         img_targets[bl] = dict(sorted(img_targets[bl].items()))
#         feats = np.array(list(img_features[bl].values()))
#         N = feats.shape[0]
#         logger.info('got image features: {} {}'.format(bl, feats.shape))
#         output = {
#             'img_features': feats.reshape(N, -1),
#             'img_inds': np.array(list(img_features[bl].keys())),
#             'img_targets': np.array(list(img_targets[bl].values())),
#         }
#         prefix = '{}_{}_'.format(opts.output_file_prefix, bl)
#         out_feat_file = os.path.join(opts.output_dir, prefix + 'features.npy')
#         out_target_file = os.path.join(opts.output_dir, prefix + 'targets.npy')
#         out_inds_file = os.path.join(opts.output_dir, prefix + 'inds.npy')
#         logger.info('Saving extracted features: {} {} to: {}'.format(
#             bl, output['img_features'].shape, out_feat_file))
#         np.save(out_feat_file, output['img_features'])
#         logger.info('Saving extracted targets: {} to: {}'.format(
#             output['img_targets'].shape, out_target_file))
#         np.save(out_target_file, output['img_targets'])
#         logger.info('Saving extracted indices: {} to: {}'.format(
#             output['img_inds'].shape, out_inds_file))
#         np.save(out_inds_file, output['img_inds'])

#     logger.info('All Done!')
#     # shut down the data loader
#     model.data_loader.shutdown_dataloader()


if __name__ == "__main__":

    _A = parser.parse_args()
    device = torch.device(f"cuda:{_A.gpu_id}" if _A.gpu_id != -1 else "cpu")

    # Configure our custom logger.
    logger.remove(0)
    logger.add(
        sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
    )

    # Print config and args.
    for arg in vars(_A):
        logger.info("{:<20}: {}".format(arg, getattr(_A, arg)))

    assert os.path.exists(_A.features_file), "Data file not found. Abort!"
    if not os.path.exists(_A.output_path):
        os.makedirs(_A.output_path)

    # -------------------------------------------------------------------------
    #   EXTRACT FEATURES FOR TRAINING SVMs
    # -------------------------------------------------------------------------


    logger.info(f"Loading pre-extracted features from {_A.features_file}...")
    features = np.load(_A.features_file, encoding="latin1").astype(np.float64)

    logger.info(f"Loading target labels from {_A.targets_file}...")
    targets = np.load(_A.targets_file, encoding="latin1")

    # Normalize image features, shape: (N, ~9000)
    features = features / (np.linalg.norm(features, axis=1) + 1e-8)[:, np.newaxis]

    # -------------------------------------------------------------------------
    #   TRAIN SVMs WITH EXTRACTED FEATURES
    # -------------------------------------------------------------------------

    # Iterate over all VOC classes and train one-vs-all linear SVMs.
    for cls_idx in range(targets.shape[1]):
        for cost_idx in range(len(_A.costs)):
            cost = _A.costs[cost_idx]
            logger.info(f"Training SVM for class {cls_idx}, cost {cost}")

            clf = LinearSVC(
                C=cost,
                class_weight={1: 2, -1: 1},
                intercept_scaling=1.0,
                verbose=1,
                penalty="l2",
                loss="squared_hinge",
                tol=0.0001,
                dual=True,
                max_iter=2000,
            )
            cls_labels = targets[:, cls_idx].astype(dtype=np.int32, copy=True)
            # meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as svm train target
            # label 1 = present. Make the svm train target labels as -1, 1.
            cls_labels[np.where(cls_labels == 0)] = -1
            num_positives = len(np.where(cls_labels == 1)[0])
            num_negatives = len(cls_labels) - num_positives
            logger.info(
                "cls: {} has +ve: {} -ve: {} ratio: {}".format(
                    cls_idx,
                    num_positives,
                    num_negatives,
                    float(num_positives) / num_negatives,
                )
            )
            logger.info(
                "features: {} cls_labels: {}".format(
                    features.shape, cls_labels.shape
                )
            )
            ap_scores = cross_val_score(
                clf, features, cls_labels, cv=3, scoring="average_precision"
            )
            clf.fit(features, cls_labels)
            logger.info(
                "cls: {} cost: {} AP: {} mean:{}".format(
                    cls_idx, cost, ap_scores, ap_scores.mean()
                )
            )

            # -----------------------------------------------------------------
            #   SAVE MODELS TO DISK
            # -----------------------------------------------------------------
            out_file = os.path.join(
                _A.output_path, f"SVM_cls_{cls_idx}_cost_{cost}.pickle"
            )
            ap_out_file = os.path.join(
                _A.output_path, f"AP_cls_{cls_idx}_cost_{cost}.npy"
            )

            logger.info(f"Saving cls cost AP to: {ap_out_file}")
            np.save(ap_out_file, np.array([ap_scores.mean()]))

            logger.info(f"Saving SVM model to: {out_file}")
            with open(out_file, "wb") as fwrite:
                pickle.dump(clf, fwrite)
