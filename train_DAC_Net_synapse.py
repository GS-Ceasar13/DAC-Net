import argparse
import logging
import os
import random
import warnings
from pydoc import locate

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer import trainer_synapse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--root_path",
    type=str,
    default="data/Synapse/train_npz",
    help="root dir for train data",
)
parser.add_argument(
    "--test_path",
    type=str,
    default="data/Synapse/test_vol_h5",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse_chongcaiyang1", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--output_dir", type=str, default="./model_out_DAC_Net", help="output dir")
# parser.add_argument(
#     "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer",default='networks.DAEFormer_original.DAEFormer'
# )

# parser.add_argument(
#     "--root_path",
#     type=str,
#     default="data/FLARE22/train_npz",
#     help="root dir for train data",
# )
# parser.add_argument(
#     "--test_path",
#     type=str,
#     default="data/FLARE22/test_vol_h5",
#     help="root dir for test data",
# )
# parser.add_argument("--dataset", type=str, default="FLARE22", help="experiment_name")
# parser.add_argument("--list_dir", type=str, default="./lists/lists_FLARE22", help="list dir")
# parser.add_argument("--num_classes", type=int, default=14, help="output channel of network")
# parser.add_argument("--model_name", type=str, default="FLARE22", help="model_name")
# parser.add_argument("--output_dir", type=str, default="./model_out_FLARE22", help="output dir")
parser.add_argument(
    "--module", help="The module that you want to load as the network, e.g. networks.DACNet",default='networks.DAC_Net.DACNet'
)


parser.add_argument("--max_iterations", type=int, default=90000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=7, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=5, help="eval_interval")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.02, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument("--ckpt", default=None, help="if or not to load net")
parser.add_argument('--latest-checkpoint-file', type=str, default="synapse_epoch_57.pth", help="Store the latest checkpoint in each epoch")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")

# parser.add_argument(
#     "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer",default='networks.DACNet-EA.DACNet'
# )

args = parser.parse_args()


if __name__ == "__main__":
    # setting device on GPU if available, else CPU
    transformer = locate(args.module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 9,
        },
        "FLARE22": {
            "root_path": args.root_path,
            "list_dir": args.list_dir,
            "num_classes": 14,
        },
    }

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = transformer(n_channels=1,n_classes=args.num_classes,bilinear=True).cuda(0)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)
        net.load_state_dict(ckpt["model"])


        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]+1
    else:
        elapsed_epochs = 0
        # elapsed_epochs = 1



    if args.resume:
        snapshot = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(snapshot):
            snapshot = snapshot.replace("best_model", "transfilm_epoch_" + str(args.max_epochs - 1))
        net.load_state_dict(torch.load(snapshot))
    trainer = {
        "Synapse": trainer_synapse,
        "FLARE22": trainer_synapse,
    }
    trainer[dataset_name](args, net, args.output_dir,elapsed_epochs)
