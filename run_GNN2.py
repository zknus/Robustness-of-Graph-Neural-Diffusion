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
from grb.model.torch import GCN, BELTRAMI, MEANCURV, pLAPLACE, HEAT, BELTRAMI2, GCNODE, GraphSAGE, GIN, APPNP
from grb.defense import RobustGCN, GCNGuard, GCNSVD
from grb.model.dgl import GAT, GATODE
from grb.utils.normalize import GCNAdjNorm, SAGEAdjNorm
from grb.trainer.trainer import Trainer
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
        attack = PGD(epsilon=.02,
                     n_epoch=args.attack_epoch,
                     n_inject_max=args.n_inject_max,
                     n_edge_max=args.n_edge_max,
                     feat_lim_min=args.feat_lim_min,
                     feat_lim_max=args.feat_lim_max,
                     device=device,
                     verbose=False)
                     
    elif attack_name == "speit":
        from grb.attack.injection.speit import SPEIT
        attack = SPEIT(lr=args.attack_lr,
                       n_epoch=args.attack_epoch,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       device=device,
                       verbose=False)
    elif attack_name == "tdgia":
        from grb.attack.injection.tdgia import TDGIA

        attack = TDGIA(lr=args.attack_lr,
                       n_epoch=args.attack_epoch,
                       n_inject_max=args.n_inject_max,
                       n_edge_max=args.n_edge_max,
                       feat_lim_min=args.feat_lim_min,
                       feat_lim_max=args.feat_lim_max,
                       device=device,
                       verbose=False)
    else:
        raise NotImplementedError

    return attack
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN models in pipeline.')
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
    parser.add_argument("--attack_epoch", type=int, default=100)
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument("--n_attack", type=int, default=1)
    parser.add_argument("--n_inject_ratio", type=float, default=None)
    parser.add_argument("--n_inject_max", type=int, default=1000)
    parser.add_argument("--n_edge_max", type=int, default=1000)
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
    
    dataset_name = 'grb-citeseer'
    
    
    if dataset_name == 'grb-pubmed':
    
        args.feat_lim_min, args.feat_lim_max = -0.143, 0.99
        args.n_inject_max, args.n_edge_max = 300, 300
        
        from grb.dataset import CustomDataset
        from scipy.sparse import csr_matrix

        data = dgl.data.PubmedGraphDataset()
        g = data[0]
        adj= g.adj_sparse('csr')

        adj_csr= csr_matrix((adj[2],adj[1],adj[0]))

        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        dataset=CustomDataset(adj=adj_csr,features=features,labels=labels,train_mask=None,val_mask=None,test_mask=None,name="pubmed",data_dir=None,mode="easy",feat_norm='arctan',save=False,verbose=True, seed=42)
        
        adj = dataset.adj
        features = dataset.features
        labels = dataset.labels
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        test_mask = dataset.test_mask
        
    elif dataset_name == 'grb-coauthor':
    
        args.feat_lim_min, args.feat_lim_max = -0.04, 1.00
        args.n_inject_max, args.n_edge_max = 150, 300
        
        from grb.dataset import CustomDataset
        from scipy.sparse import csr_matrix

        data = dgl.data.CoauthorCSDataset()
        g = data[0]
        adj= g.adj_sparse('csr')

        adj_csr= csr_matrix((adj[2],adj[1],adj[0]))

        features = g.ndata['feat']
        labels = g.ndata['label']

        dataset=CustomDataset(adj=adj_csr,features=features,labels=labels,train_mask=None,val_mask=None,test_mask=None,name="coauthor",data_dir=None,mode="easy",feat_norm='arctan',save=False,verbose=True, seed=42)
        
        adj = dataset.adj
        features = dataset.features
        labels = dataset.labels
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        test_mask = dataset.test_mask
        
    elif dataset_name == 'grb-AmazonCoBuyComputerDataset':
    
        args.feat_lim_min, args.feat_lim_max = -0.402, 0.598
        args.n_inject_max, args.n_edge_max = 100, 200
        
        from grb.dataset import CustomDataset
        from scipy.sparse import csr_matrix

        data = dgl.data.AmazonCoBuyComputerDataset()
        g = data[0]
        adj= g.adj_sparse('csr')
        adj_dense=g.adj()
        adj_csr= csr_matrix((adj[2],adj[1],adj[0]),shape=(adj_dense.shape[0],adj_dense.shape[1]))

        features = g.ndata['feat']
        labels = g.ndata['label']

        dataset=CustomDataset(adj=adj_csr,features=features,labels=labels,train_mask=None,val_mask=None,test_mask=None,name="amazoncobuyComputer",data_dir=None,mode="easy",feat_norm='arctan',save=False,verbose=True, seed=42)

        adj = dataset.adj
        features = dataset.features
        labels = dataset.labels
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        test_mask = dataset.test_mask
    
    elif dataset_name == 'grb-ogbn-arxiv':
        from grb.dataset import OGBDataset
        
        args.n_inject_max, args.n_edge_max = 1000, 5000
        
        dataset = OGBDataset(name='ogbn-arxiv', data_dir='./data')

        args.feat_lim_min, args.feat_lim_max = -1.3889, 1.6387
        adj = dataset.adj
        features = dataset.features
        labels = dataset.labels
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        test_mask = dataset.test_mask
        
    else:

        if dataset_name == 'grb-cora':
            args.feat_lim_min, args.feat_lim_max = -0.94, 0.94
            args.n_inject_max, args.n_edge_max = 50, 50
        elif dataset_name == 'grb-citeseer':
            args.feat_lim_min, args.feat_lim_max = -0.96, 0.89
            args.n_inject_max, args.n_edge_max = 50, 50
        elif dataset_name == 'grb-flickr':
            args.feat_lim_min, args.feat_lim_max = -0.47, 1.00
            args.n_inject_max, args.n_edge_max = 1000, 5000
        elif dataset_name == 'grb-reddit':
            args.feat_lim_min, args.feat_lim_max = -0.98, 0.99
            
        dataset = Dataset(name=dataset_name, mode='easy', feat_norm='arctan')
                          
        adj = dataset.adj
        features = dataset.features
        labels = dataset.labels
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        test_mask = dataset.test_mask

    # ----------Train surrogate model--------------
    model_name = "gcn"
    model_sur = GCN(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64, 
                    n_layers=3,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=False,
                    residual=False,
                    dropout=0.5)
                    

    save_dir = "./saved_surmodels/{}/{}".format(dataset_name, model_name)
    save_name = "gcn_sur.pt"
    device = "cuda:0"
    feat_norm = None
    train_mode = "inductive"
    
    trainer = Trainer(dataset=dataset, 
                        optimizer=torch.optim.Adam(model_sur.parameters(), lr=0.01),
                        loss=torch.nn.functional.cross_entropy,
                        lr_scheduler=False,
                        early_stop=True,
                        early_stop_patience=500,
                        feat_norm=feat_norm,
                        device=device)
    trainer.train(model=model_sur, 
                        n_epoch=2000,
                        eval_every=1,
                        save_after=0,
                        save_dir=save_dir,
                        save_name=save_name,
                        train_mode=train_mode,
                        verbose=False)
    
    
    ckp = torch.load(os.path.join(save_dir, save_name), map_location=device)
    model_sur.load_state_dict(ckp['model']) 
              
    # by trainer
    test_score = trainer.evaluate(model_sur, dataset.test_mask)
    print("Test score of surrogate model: {:.4f}".format(test_score))
    
    
    # ----------Attack surrogate model--------------
    attack = build_attack_adv("speit", args=args)
    adj_attack, features_attack = attack.attack(model=model_sur, adj=adj, features=features, target_mask=test_mask, adj_norm_func=model_sur.adj_norm_func)
    features_attacked = torch.cat([features.to(device), features_attack.to(device)])
    test_score = utils.evaluate(model_sur, 
                                features=features_attacked,
                                adj=adj_attack,
                                labels=dataset.labels,
                                adj_norm_func=model_sur.adj_norm_func,
                                mask=dataset.test_mask,
                                device=device)
    print("Test score after attack for surrogate model: {:.4f}.".format(test_score))
    
    del model_sur, trainer
    
    # from grb.trainer.trainer2 import Trainer
    # ----------Adv train target model--------------
    save_name = args.save_name.split('.')[0] + "_{}.pt".format(0)
    p = 1 # p-norm for pLaplace flow, p=1,2 corresponding to p=3,4 in the paper (supplementary)
    model_name = "beltrami_1noAdvT_drop05_attsamp095"
    #model_name = "gat_ln_noAdvT" # BeltramiGuard
    
    args.save_dir = "./saved_models2/{}/{}".format(dataset_name, model_name)
    
    if model_name.split('_')[0] == "beltrami":
        model = BELTRAMI(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        adj_norm_func=GCNAdjNorm,
                        layer_norm=True,
                        residual=False,
                        dropout=0.5)

    if model_name.split('_')[0] == "beltrami2": # removed rowsum
        model = BELTRAMI2(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        adj_norm_func=GCNAdjNorm,
                        layer_norm=True,
                        residual=False,
                        dropout=0.5)
                        
    if model_name.split('_')[0] == "pLaplace":
        model = pLAPLACE(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        p=p,
                        adj_norm_func=GCNAdjNorm,
                        layer_norm=True,
                        residual=False,
                        dropout=0.5)
                        
    if model_name.split('_')[0] == "meancurv":
        model = MEANCURV(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64, 
                        n_layers=3,
                        adj_norm_func=GCNAdjNorm,
                        layer_norm=True,
                        residual=False,
                        dropout=0.5)
                        
    if model_name.split('_')[0] == "heat":
        model = HEAT(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        adj_norm_func=GCNAdjNorm,
                        layer_norm=True,
                        residual=False,
                        dropout=0.5)
                        
    if model_name.split('_')[0] == "grand":
        model = GCNODE(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        adj_norm_func=GCNAdjNorm,
                        layer_norm=True,
                        residual=False,
                        dropout=0.5)
                        
    elif model_name.split('_')[0] == "robustgcn":           
        model = RobustGCN(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        dropout=0.5)                    
    
    elif model_name.split('_')[0] == "gcnguard":            
        model = GCNGuard(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        layer_norm=True,
                        n_layers=3,
                        dropout=0.5)    

    elif model_name.split('_')[0] == "gcnsvd":  
        model = GCNSVD(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        layer_norm=True,
                        adj_norm_func=GCNAdjNorm,
                        n_layers=3,
                        dropout=0.5)  
                        
    elif model_name.split('_')[0] == "gat":            
        model = GAT(in_features=dataset.num_features,
                        out_features=dataset.num_classes,
                        hidden_features=64,
                        n_layers=3,
                        n_heads=4,
                        layer_norm=True,
                        dropout=0.5)
                        
    elif model_name.split('_')[0] == "graphsage":                   
        model = GraphSAGE(in_features=dataset.num_features,
                      out_features=dataset.num_classes,
                      hidden_features=64,
                      n_layers=3,
                      adj_norm_func=SAGEAdjNorm,
                      layer_norm=True,
                      dropout=0.5)
                      
    elif model_name.split('_')[0] == "gin":                     
        model = GIN(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64, 
                    n_layers=3,
                    adj_norm_func=None,
                    layer_norm=True,
                    batch_norm=True,
                    dropout=0.5)
                    
    elif model_name.split('_')[0] == "appnp":                   
        model = APPNP(in_features=dataset.num_features,
                      out_features=dataset.num_classes,
                      hidden_features=64, 
                      n_layers=3,
                      adj_norm_func=GCNAdjNorm,
                      layer_norm=True,
                      edge_drop=0.1,
                      alpha=0.01,
                      k=3,
                      dropout=0.5)
              
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
                  save_dir=args.save_dir,
                  save_name=save_name,
                  eval_every=args.eval_every,
                  save_after=args.save_after,
                  train_mode=train_params["train_mode"] if "train_mode" in train_params else args.train_mode,
                  verbose=args.verbose)
              

    ckp = torch.load(os.path.join(args.save_dir, save_name), map_location=device)
    model.load_state_dict(ckp['model'])  
    val_score = trainer.evaluate(model, dataset.val_mask)
    test_score = trainer.evaluate(model, dataset.test_mask)

    print("*" * 80)
    print("Val ACC of {}: {:.4f}".format(model_name, val_score))
    print("Test ACC of {}: {:.4f}".format(model_name, test_score))


    print("Adversarial training finished.")


    # ----------Attack target model--------------

    model.eval()
    test_score = utils.evaluate(model, 
                            features=features_attacked,
                            adj=adj_attack,
                            labels=dataset.labels,
                            adj_norm_func=model.adj_norm_func,
                            mask=dataset.test_mask,
                            device=device)
    print("Test score after attack for target model: {:.4f}.".format(test_score))
    
    del model, trainer