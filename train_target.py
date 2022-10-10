import torch
from torch import nn
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch_geometric.nn.conv import MessagePassing
from utils import Meter
from torchdiffeq import odeint
import argparse
import torch_sparse

from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from utils import get_rw_adj, gcn_norm_fill_val
import dgl
import os
import torch
import grb.utils as utils
from grb.dataset import Dataset
from grb.model.torch import GCN, GCNODE
from grb.model.dgl import GAT, GATODE
from grb.utils.normalize import GCNAdjNorm
from grb.trainer.trainer import Trainer
from grb.attack.injection.tdgia import TDGIA
from grb.attack.injection.speit import SPEIT
from grb.defense import AdvTrainer
from torch_geometric.utils import softmax
import numpy as np

from torch_geometric.utils.loop import add_remaining_self_loops

device = torch.device('cuda')
    
def build_attack_adv(attack_name, device="cpu", args=None):
    if attack_name == "rand":
        from grb.attack.injection import RAND

        attack = RAND(n_inject_max=args.n_inject_max,
                      n_edge_max=args.n_edge_max,
                      feat_lim_min=args.feat_lim_min,
                      feat_lim_max=args.feat_lim_max,
                      device=device,
                      verbose=False)
    elif attack_name == "fgsm":
        from grb.attack.injection import FGSM

        attack = FGSM(epsilon=args.attack_lr,
                      n_epoch=args.attack_epoch,
                      n_inject_max=args.n_inject_max,
                      n_edge_max=args.n_edge_max,
                      feat_lim_min=args.feat_lim_min,
                      feat_lim_max=args.feat_lim_max,
                      device=device,
                      verbose=False)
    elif attack_name == "pgd":
        from grb.attack.injection import PGD

        attack = PGD(epsilon=args.lr,
                     n_epoch=args.attack_epoch,
                     n_inject_max=args.n_inject_max,
                     n_edge_max=args.n_edge_max,
                     feat_lim_min=args.feat_lim_min,
                     feat_lim_max=args.feat_lim_max,
                     device=device,
                     verbose=False)
    elif attack_name == "speit":
        from grb.attack.injection import SPEIT

        attack = SPEIT(lr=args.attack_lr,
                       n_epoch=args.attack_epoch,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       device=device,
                       verbose=False)
    elif attack_name == "tdgia":
        from grb.attack.injection import TDGIA

        attack = TDGIA(lr=args.attack_lr,
                       n_epoch=1000,
                       n_inject_max=60,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       device=device,
                       verbose=False)
    else:
        raise NotImplementedError

    return attack
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial training GNN models in pipeline.')
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="grb-cora")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--feat_norm", type=str, default="arctan")
    # Model settings
    parser.add_argument("--model", nargs='+', default=None)
    parser.add_argument("--save_dir", type=str, default="./saved_models2/")
    parser.add_argument("--config_dir", type=str, default="./pipeline/configs/")
    parser.add_argument("--log_dir", type=str, default="./pipeline/logs/")
    parser.add_argument("--save_name", type=str, default="model_at.pt")
    # Attack setting
    parser.add_argument("--attack_adv", type=str, default="fgsm")
    parser.add_argument("--attack_epoch", type=int, default=10)
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument("--n_attack", type=int, default=1)
    parser.add_argument("--n_inject_ratio", type=float, default=None)
    parser.add_argument("--n_inject_max", type=int, default=20)
    parser.add_argument("--n_edge_max", type=int, default=20)
    parser.add_argument("--feat_lim_min", type=float, default=-.94)
    parser.add_argument("--feat_lim_max", type=float, default=.94)
    # Adversarial training settings
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_train", type=int, default=1)
    parser.add_argument("--n_epoch", type=int, default=8000, help="Training epoch.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_after", type=int, default=0)
    parser.add_argument("--train_mode", type=str, default="inductive")
    parser.add_argument("--eval_metric", type=str, default="acc")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=500)
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    dataset_name = 'grb-cora'
    dataset = Dataset(name='grb-cora', mode='easy', feat_norm='arctan')
                      
    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    test_mask = dataset.test_mask
    

    # ----------Adv train target model--------------
    save_name = args.save_name.split('.')[0] + "_{}.pt".format(0)
    model_name = "grand3_noAdvT_drop05_attsamp095"
    # model_name = "gat4_noAdvT"
        
    model = GCNODE(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64, 
                    n_layers=3,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=True,
                    residual=False,
                    dropout=0.5)
                
    # model = GAT(in_features=num_features,
                    # out_features=num_classes,
                    # hidden_features=64,
                    # n_layers=3,
                    # n_heads=4,
                    # layer_norm=True if "ln" in model_name else False,
                    # dropout=0.5)
                    
    train_params = {
        "lr": 0.001,
        "n_epoch": 5000,
        "early_stop": True,
        "early_stop_patience": 500,
        "train_mode": "inductive",
    }
    
    import config
    
    optimizer = config.build_optimizer(model=model, lr=args.lr)
    loss = config.build_loss()
    eval_metric = config.build_metric()
    
    # attack_adv = build_attack_adv(args.attack_adv, device=device, args=args)
    
    # trainer = AdvTrainer(dataset=dataset,
                         # optimizer=optimizer,
                         # loss=loss,
                         # attack=attack_adv,
                         # lr_scheduler=args.lr_scheduler,
                         # eval_metric=eval_metric,
                         # early_stop=train_params[
                             # "early_stop"] if "early_stop" in train_params else args.early_stop,
                         # early_stop_patience=train_params[
                             # "early_stop_patience"] if "early_stop_patience" in train_params else args.early_stop_patience,
                         # device=device)
                         
    trainer = Trainer(dataset=dataset, 
                      optimizer=optimizer,
                      loss=loss,
                      lr_scheduler=args.lr_scheduler,
                      early_stop=train_params[
                             "early_stop"] if "early_stop" in train_params else args.early_stop,
                      early_stop_patience=train_params[
                             "early_stop_patience"] if "early_stop_patience" in train_params else args.early_stop_patience,
                      device=device)
                      
    trainer.train(model=model,
                  n_epoch=train_params["n_epoch"] if "n_epoch" in train_params else args.n_epoch,
                  save_dir=os.path.join(args.save_dir, model_name),
                  save_name=save_name,
                  eval_every=args.eval_every,
                  save_after=args.save_after,
                  train_mode=train_params["train_mode"] if "train_mode" in train_params else args.train_mode,
                  verbose=args.verbose)
              

    #model.dropout=0.0
    ckp = torch.load(os.path.join(args.save_dir, model_name, save_name), map_location=device)
    model.load_state_dict(ckp['model'])  
    val_score = trainer.evaluate(model, dataset.val_mask)
    test_score = trainer.evaluate(model, dataset.test_mask)

    print("*" * 80)
    print("Val ACC of {}: {:.4f}".format(model_name, val_score))
    print("Test ACC of {}: {:.4f}".format(model_name, test_score))


    print("Adversarial training finished.")

