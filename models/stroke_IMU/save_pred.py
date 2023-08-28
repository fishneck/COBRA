import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.class_id_map import get_n_classes
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.postprocess import PostProcessor
from libs.transformer import TempDownSamp, ToTensor
import time
import pickle
def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="save predictions from Action Segmentation Networks."
    )
    parser.add_argument(
        "config",
        type=str,
        help="path to a config file about the experiment on action segmentation",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Add --cpu option if you use cpu."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="path to a config file about the experiment on action segmentation",
        default = 'cer',
    )

    return parser.parse_args()


def predict(
    loader: DataLoader,
    model: nn.Module,
    device: str,
    result_path: str,
    boundary_th: float,
    model_type:str,
    stroke = False,
) -> None:
    if model_type == 'cer':
        save_dir = os.path.join(result_path, "predictions_cer_model.p")
    elif model_type == 'acc':
        save_dir = os.path.join(result_path, "predictions_acc_model.p")
    elif model_type == 'edit':
        save_dir = os.path.join(result_path, "predictions_edit_model.p")
    elif model_type == 'loss':
        save_dir = os.path.join(result_path, "predictions_loss_model.p")
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#         print("directory created")

    postprocessor = PostProcessor("refinement_with_boundary", boundary_th)

    # switch to evaluate mode
    model.eval()
    all_prob = []
    all_r_pred = []
    all_gt = []
    all_name = []
    with torch.no_grad():
        for sample in loader:
            x = sample["feature"]
            t = sample["label"]
            path = sample["feature_path"][0]
            name = os.path.basename(path)
            mask = sample["mask"].numpy()

            x = x.to(device)

            # compute output and loss
            output_cls, output_bound = model(x)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            refined_pred = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            pred = output_cls.argmax(axis=1)
#             np.save(os.path.join(save_dir, name[:-4] + "_pred_prob.npy"),output_cls)
#             np.save(os.path.join(save_dir, name[:-4] + "_pred.npy"), pred[0])
#             np.save(
#                 os.path.join(save_dir, name[:-4] + "_refined_pred.npy"), refined_pred[0]
#             )
#             np.save(os.path.join(save_dir, name[:-4] + "_gt.npy"), t[0])
            all_prob.append(output_cls)
            all_r_pred.append(refined_pred[0])
            all_gt.append(t[0])
            all_name.append(name[:-4])
    pred_dict = {'probs':all_prob,'r_pred':all_r_pred,'gt':all_gt,'name':all_name}
    if stroke:
        pickling(pred_dict,save_dir[:-1]+'_stroke.p')
    else:
        pickling(pred_dict,save_dir)

            # make graph for boundary regression
#             output_bound = output_bound[0, 0]
#             h_axis = np.arange(len(output_bound))
#             fig = plt.figure()
#             ax = fig.add_subplot(1, 1, 1)
#             plt.tick_params(labelbottom=False, labelright=False, labeltop=False)
#             plt.ylim(0.0, 1.0)
#             ax.set_yticks([0, boundary_th, 1])
#             ax.spines["right"].set_color("none")
#             ax.spines["left"].set_color("none")
#             ax.plot(h_axis, output_bound, color="#e46409")
#             plt.savefig(os.path.join(save_dir, name[:-4] + "_boundary.png"))
#             plt.close(fig)


def main():
    start_time = time.time()
    args = get_arguments()

    # configuration
    config = get_config(args.config)
    result_path = os.path.dirname(args.config)

    # cpu or gpu
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True

    # Dataloader
    downsamp_rate = 2 if config.dataset == "50salads" else 1

    data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="test",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    
    s_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="s_test",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    s_loader = DataLoader(
        s_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    
    train_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="training",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    
    val_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="validation",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)

    model = models.ActionSegmentRefinementFramework(
        in_channel=config.in_channel,
        n_features=config.n_features,
        n_classes=n_classes,
        n_stages=config.n_stages,
        n_layers=config.n_layers,
        n_stages_asb=config.n_stages_asb,
        n_stages_brb=config.n_stages_brb,
    )

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model_type == 'cer':
        state_dict_cls = torch.load(os.path.join(result_path, "best_cer_model.prm"))
    elif args.model_type == 'acc':
        state_dict_cls = torch.load(os.path.join(result_path, "best_acc_model.prm"))
    elif args.model_type == 'edit':
        state_dict_cls = torch.load(os.path.join(result_path, "best_edit_model.prm"))
    elif args.model_type == 'loss':
        state_dict_cls = torch.load(os.path.join(result_path, "best_loss_model.prm"))
    model.load_state_dict(state_dict_cls)

    # save outputs
    predict(loader, model, device, result_path, config.boundary_th,args.model_type)
    predict(s_loader, model, device, result_path, config.boundary_th,args.model_type, stroke=True)
#     predict(train_loader, model, device, result_path, config.boundary_th,args.model_type)
#     predict(val_loader, model, device, result_path, config.boundary_th,args.model_type)
    end_time = time.time()
    print('Total time taken to perform prediction on the entire dataset is {} mins.'.format((end_time-start_time)/60))
    print("Done")


if __name__ == "__main__":
    main()
