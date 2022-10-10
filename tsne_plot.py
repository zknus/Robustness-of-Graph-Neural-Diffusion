import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
from dgl import DGLGraph
from grb.dataset import Dataset
from grb.utils.normalize import GCNAdjNorm
from grb.utils import utils
from grb.model.torch import GCN, GCNODE, GCNODE2, BELTRAMI, MEANCURV, pLAPLACE, HEAT
import os
import numpy as np
import scipy
from torch_geometric.utils import remove_self_loops
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix

device = torch.device('cuda')

def plot(g, attention, ax, nodes_to_plot=None, nodes_labels=None,
         edges_to_plot=None, nodes_pos=None, nodes_colors=None,
         edge_colormap=plt.cm.Greys):
   
    if nodes_to_plot is None:
        nodes_to_plot = g.nodes()
    if edges_to_plot is None:
        assert isinstance(g, nx.DiGraph), 'Expected g to be an networkx.DiGraph' \
                                          'object, got {}.'.format(type(g))
        edges_to_plot = sorted(g.edges())
    nx.draw_networkx_edges(g, nodes_pos, edgelist=edges_to_plot,
                           edge_color=attention, edge_cmap=edge_colormap,
                           width=1, alpha=0.5, ax=ax, edge_vmin=0,
                           edge_vmax=1)

    if nodes_colors is None:
        nodes_colors = sns.color_palette("deep", max(nodes_labels) + 1)

    nx.draw_networkx_nodes(g, nodes_pos, nodelist=nodes_to_plot, ax=ax, node_size=10,
                           node_color=[nodes_colors[nodes_labels[v]] for v in nodes_to_plot],
                           alpha=0.9)
                           
                           
dataset_name = 'grb-cora'
dataset = Dataset(name='grb-cora', mode='easy', feat_norm='arctan')
                  
adj = dataset.adj
features = dataset.features
labels = dataset.labels
num_features = dataset.num_features
num_classes = dataset.num_classes
mask = dataset.train_mask


save_name = "model_at.pt"
save_name = save_name.split('.')[0] + "_{}.pt".format(0)
model_name = "meancurv_noAdvT_drop05_attsamp095"

save_dir = "./saved_models2/{}/{}".format(dataset_name, model_name)

if model_name.split('_')[0] == "beltrami":
    model = BELTRAMI(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=128,
                    n_layers=4,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=True,
                    residual=False,
                    dropout=0.5)
                    
if model_name.split('_')[0] == "meancurv":
    model = MEANCURV(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=128,
                    n_layers=4,
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
                    
if model_name.split('_')[0] == "grand2":
    model = GCNODE2(in_features=dataset.num_features,
                    out_features=dataset.num_classes,
                    hidden_features=64,
                    n_layers=3,
                    adj_norm_func=GCNAdjNorm,
                    layer_norm=True,
                    residual=False,
                    dropout=0.5)
                        
ckp = torch.load(os.path.join(save_dir, save_name), map_location=device)
model.load_state_dict(ckp['model'])
model.to(device)
model.eval()

adj_ = utils.adj_preprocess(adj, adj_norm_func=model.adj_norm_func, mask=mask, model_type=model.model_type)
logits, att, edge_index = model(features[mask].to(device), adj_.to(device))

t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(logits.cpu().detach().numpy())
pos={}
for i in range(logits.shape[0]):
    x = t_sne_embeddings[i,0]
    y = t_sne_embeddings[i,1]
    pos[i] = (x,y)
    
# node_labels = labels[mask]
# cora_label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
# fig = plt.figure(figsize=(12,8), dpi=80)  # otherwise plots are really small in Jupyter Notebook
# for class_id in range(num_classes):
    # plt.scatter(t_sne_embeddings[node_labels == class_id, 0], t_sne_embeddings[node_labels == class_id, 1], s=20, color=cora_label_to_color_map[class_id], edgecolors='black', linewidths=0.2)
# plt.show()
# plt.savefig('b.png')

# mean_att = 0
# for i in range(len(att)):
    # mean_att += att[i]
# mean_att /= len(att)


# adj2 = adj_.to_dense().cpu().detach().numpy()
# adj2 -= np.diag(np.diag(adj2))

#src, dst = np.nonzero(adj2)


ns = list(range(mask[mask==True].shape[0]))
for layer in range(1):
    print(layer, len(att), edge_index[layer])
    edge_index_noself, new_weights = remove_self_loops(edge_index[layer], att[layer])

    src = edge_index_noself[0].cpu().detach().numpy()
    dst = edge_index_noself[1].cpu().detach().numpy()
    g_dgl = DGLGraph((src, dst)).cpu()
    dg = g_dgl.to_networkx().to_directed()
    dg.edges()
    

    a = csr_matrix((new_weights.cpu().detach().numpy(), (src, dst)), shape=(len(ns), len(ns)))
    b = a.toarray()
    
    U, S, Vh = np.linalg.svd(b, full_matrices=False)
    plt.plot(list(range(len(S))), S)
    plt.show()
    plt.savefig(model_name+'_layer'+str(layer)+'_svd.png')
    plt.close()

    ent = []
    for i in range(len(ns)):
        p = att[layer][edge_index[layer][0]==i].cpu().detach().numpy()
        print(np.sum(p))
        if p.shape[0] == 0:
            ent.append(0)
        else:
            a = np.ones((p.shape[0], 1))/p.shape[0]
            z = np.sum(-np.log(a)*a)
            ent.append(np.sum(-np.log(p)*p)/z)
    plt.plot(ns, ent)
    plt.show()
    plt.savefig(model_name+'_layer'+str(layer)+'.png')
    plt.close()
    
    #Y.reshape(Y.shape[0])
    grEd = dg.edges()
    att_list = new_weights.cpu().detach().numpy().tolist()


    fig, ax = plt.subplots()
    plot(dg, att_list, ax=ax, nodes_pos=pos, nodes_labels=labels[mask])
    # ax.set_axis_off()
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # plt.colorbar(sm, fraction=0.046, pad=0.01)
    plt.show()
    plt.savefig(model_name+'_layer'+str(layer)+'_tsne.png')
    plt.close()