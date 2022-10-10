"""Torch module for GCN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grb.utils.normalize import GCNAdjNorm
from torch_geometric.nn.conv import MessagePassing
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from torch.nn.utils import spectral_norm
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import to_undirected
import geotorch
import numpy as np
import torch_sparse
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops
from utils import get_rw_adj
import scipy.sparse as sp


device = torch.device('cuda')

def feature_smoothing(adj, X):
    XLXT = torch.matmul(torch.matmul(X.t(), adj), X)
    loss_smooth_feat = torch.trace(XLXT)/adj.shape[0]
    return loss_smooth_feat

class ODEfunc_feat(nn.Module):
    def __init__(self, dim):
        super(ODEfunc_feat, self).__init__()
        self.fc = nn.Linear(dim, dim)
        geotorch.positive_definite(self.fc, "weight")
    def forward(self, t, x): 
        x = -self.fc(x)
        return x

class ODEBlock_feat(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock_feat, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

class ODEblock(nn.Module):
  def __init__(self, odefunc, in_features, out_features, adj, t):
    super(ODEblock, self).__init__()
    self.t = t
    
    self.aug_dim = 1 
    self.hidden_dim = 16
    self.odefunc = odefunc(in_features, out_features, adj)
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self):
    super(ODEFunc, self).__init__()
    self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None

  def __repr__(self):
    return self.__class__.__name__

class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, adj):
    super(LaplacianODEFunc, self).__init__()
    #self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.hidden_dim = 64
    #self.linear = nn.Linear(self.hidden_dim, self.hidden_dim).to(device)
    self.adj = adj

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    ax = torch.sparse.mm(self.adj, x)
    alpha = torch.sigmoid(self.alpha_train)
    f = ax - x
    return f

class ODEFuncTransformerAtt(ODEFunc):

  def __init__(self, in_features, out_features, adj):
    super(ODEFuncTransformerAtt, self).__init__()
    
    self.edge_index1 = adj._indices().to(device)
    self.edge_attr = adj._values().to(device)
    self.edge_index, self.edge_weight = add_remaining_self_loops(self.edge_index1, self.edge_attr, fill_value=1)
    
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, edge_weights=self.edge_weight).to(device)
    
  def multiply_attention(self, x, attention, v=None):
    num_heads = 4
    mix_features = 0
    if mix_features:
        vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(num_heads)], dim=0),
        dim=0)
        ax = self.multihead_att_layer.Wout(vx)
    else:
        mean_attention = attention.mean(dim=1)
        
        grad_x = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x) - x
        grad_x_abs = torch.abs(grad_x)
        grad_x_norm = torch.sqrt(torch.sum(torch.clamp(grad_x_abs*grad_x_abs, min=1e-1), 1))
        grad_x_norm_inv = 1/grad_x_norm
        attention2 = grad_x_norm_inv[self.edge_index[0,:]] + grad_x_norm_inv[self.edge_index[1,:]]
        new_attn = mean_attention * softmax(attention2, self.edge_index[0])
        rowsum_norm = 1
        if rowsum_norm:
            #Da = torch.diag(grad_x_norm_inv)
            W = torch.sparse.FloatTensor(self.edge_index, new_attn, (x.shape[0], x.shape[0])).coalesce()
            rowsum = torch.sparse.mm(W, torch.ones((W.shape[0], 1), device=device)).flatten()
            
            ni = torch.arange(x.shape[0])
            diag_index = torch.stack((ni, ni)).to(device)
            dx = torch_sparse.spmm(diag_index, rowsum, x.shape[0], x.shape[0], x)
            ax = torch_sparse.spmm(self.edge_index, new_attn, x.shape[0], x.shape[0], x)
        else:
            dx = x
            ax = torch_sparse.spmm(self.edge_index, new_attn, x.shape[0], x.shape[0], x)
    return ax-dx

  def forward(self, t, x):  # t is needed when called by the integrator
    
    attention, values = self.multihead_att_layer(x, self.edge_index)
    f = self.multiply_attention(x, attention, values)
    
    #f = (ax - x)
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = .2
    self.device = device
    self.h = int(4) # num of attention heads
    self.edge_weights = edge_weights
    self.reweight_attention = 0
    self.sn = True
    
    try:
      self.attention_dim = 64
    except KeyError:
      self.attention_dim = self.out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    if self.sn == True:
        self.Q = spectral_norm(nn.Linear(self.in_features, self.attention_dim))
        self.init_weights(self.Q)
        self.V = spectral_norm(nn.Linear(self.in_features, self.attention_dim))
        self.init_weights(self.V)
        self.K = spectral_norm(nn.Linear(self.in_features, self.attention_dim))
        self.init_weights(self.K)
    else:
        self.Q = nn.Linear(self.in_features, self.attention_dim)
        self.init_weights(self.Q)
        self.V = nn.Linear(self.in_features, self.attention_dim)
        self.init_weights(self.V)
        self.K = nn.Linear(self.in_features, self.attention_dim)
        self.init_weights(self.K)
    

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)
    self.Wout = spectral_norm(nn.Linear(self.d_k, in_features))
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):
    """
    x might be [features, augmentation, positional encoding, labels]
    """
    q = self.Q(x)
    k = self.K(x)
    v = self.V(x)

    # perform linear operation and split into h heads

    k = k.view(-1, self.h, self.d_k)
    q = q.view(-1, self.h, self.d_k)
    v = v.view(-1, self.h, self.d_k)

    # transpose to get dimensions [n_nodes, attention_dim, n_heads]

    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)

    src = q[edge[0, :], :, :]
    dst_k = k[edge[1, :], :, :]

    prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
    if self.reweight_attention and self.edge_weights is not None:
        prods = prods * self.edge_weights.unsqueeze(dim=1)
    attention = softmax(prods, edge[0])
    
    return attention, v#(v, prods)

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class AttODEblock(ODEblock):
  def __init__(self, odefunc, in_features, out_features, adj, t=torch.tensor([0, 1])):
    super(AttODEblock, self).__init__(odefunc, in_features, out_features, adj, t)

    self.odefunc = odefunc(in_features, out_features, adj)
    self.odefunc.edge_index = adj._indices().to(device)
    self.odefunc.edge_attr = adj._values().to(device)
    
    self.adjoint = True
    
    if self.adjoint:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()
    # parameter trading off between attention and the Laplacian
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features).to(device)

  def get_attention_weights(self, x):
    attention, values = self.multihead_att_layer(x, self.odefunc.edge_index)
    return attention

  def forward(self, x):
    t = self.t.type_as(x)
    self.odefunc.attention_weights = self.get_attention_weights(x)
    integrator = self.train_integrator if self.training else self.test_integrator

    func = self.odefunc
    state = x
    
    if self.adjoint:
        state_dt = integrator(func, state, t, method='dopri5',
        options={'step_size': 1},
        adjoint_method='adaptive_heun',
        adjoint_options={'step_size': 1},
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
        # state_dt = integrator(func, state, t, method='dopri5')
        state_dt = integrator(func, state, t, method='implicit_adams', options=dict(step_size=2))
    z = state_dt[1]
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"

class RewireAttODEblock(ODEblock):
  def __init__(self, odefunc, in_features, out_features, adj, t=torch.tensor([0, 1]), training=True):
    super(RewireAttODEblock, self).__init__(odefunc, in_features, out_features, adj, t)
    self.odefunc = odefunc(in_features, out_features, adj)
    self.num_nodes = int(adj.size()[0])
    edge_index = adj._indices()
    edge_attr = adj._values()
    edge_index, edge_weight = get_rw_adj(edge_index, edge_weight=edge_attr, norm_dim=1,
                                         fill_value=1,
                                         num_nodes=self.num_nodes,
                                         dtype=edge_attr.dtype)
    self.training = training
    if self.training:
        self.dropedge_perc = .5
        if self.dropedge_perc < 1:
            nnz = len(edge_attr)
            perm = np.random.permutation(nnz)
            preserve_nnz = int(nnz*self.dropedge_perc)
            perm = perm[:preserve_nnz]
            edge_weight = edge_weight[perm]
            edge_index = edge_index[:,perm]  
    else:
        self.dropedge_perc = 1
        if self.dropedge_perc < 1:
            nnz = len(edge_attr)
            perm = np.random.permutation(nnz)
            preserve_nnz = int(nnz*self.dropedge_perc)
            perm = perm[:preserve_nnz]
            edge_weight = edge_weight[perm]
            edge_index = edge_index[:,perm] 
        
    self.data_edge_index = edge_index.to(device)
    self.odefunc.edge_index = edge_index.to(device)  # this will be changed by attention scores
    self.odefunc.edge_weight = edge_weight.to(device)
    self.rw_addD = 0.02
    
    self.adjoint = False
    
    if self.adjoint:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint
      
    #from torchdiffeq import odeint_adjoint as odeint
    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()

    #self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features).to(device)

  def get_attention_weights(self, x):
    attention, values = self.odefunc.multihead_att_layer(x, self.data_edge_index)
    # attention, values = self.multihead_att_layer(x, self.data_edge_index)
    return attention

  def renormalise_attention(self, attention):
    attention_norm_idx = 0
    index = self.odefunc.edge_index[attention_norm_idx]
    att_sums = scatter(attention, index, dim=0, dim_size=self.num_nodes, reduce='sum')[index]
    return attention / (att_sums + 1e-16)


  def add_random_edges(self):
    # M = self.opt["M_nodes"]
    # M = int(self.num_nodes * (1/(1 - (1 - self.opt['att_samp_pct'])) - 1))
    M = int(self.num_nodes * (1/(1 - self.rw_addD) - 1))

    with torch.no_grad():
      new_edges = np.random.choice(self.num_nodes, size=(2,M), replace=True, p=None)
      new_edges = torch.tensor(new_edges).to(device)
      #todo check if should be using coalesce insted of unique
      #eg https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/two_hop.html#TwoHop
      cat = torch.cat([self.data_edge_index, new_edges],dim=1)
      no_repeats = torch.unique(cat, sorted=False, return_inverse=False,
                                return_counts=False, dim=1)
      self.data_edge_index = no_repeats
      self.odefunc.edge_index = self.data_edge_index

  def add_khop_edges(self, k=2, rm_self_loops=True):
    n = self.num_nodes
    for i in range(k-1):
      new_edges, new_weights = torch_sparse.spspmm(self.odefunc.edge_index, self.odefunc.edge_weight,
                            self.odefunc.edge_index, self.odefunc.edge_weight, n, n, n, coalesced=True)

      new_edges, new_weights = remove_self_loops(new_edges, new_weights)

      # A1 = torch.sparse_coo_tensor(self.odefunc.edge_index, self.odefunc.edge_weight, (n, n)).coalesce()
      # A2 = torch.sparse_coo_tensor(new_edges, new_weights, (n, n)).coalesce()

      A1pA2_index = torch.cat([self.odefunc.edge_index, new_edges], dim=1)
      A1pA2_value = torch.cat([self.odefunc.edge_weight, new_weights], dim=0) / 2
      ei, ew = torch_sparse.coalesce(A1pA2_index, A1pA2_value, n, n, op="add")

      self.data_edge_index = ei
      self.odefunc.edge_index = self.data_edge_index
      self.odefunc.attention_weights = ew

  def densify_edges(self):
    new_edges = 'random' # 'random_walk' or 'k_hop_lap' or 'k_hop_att'
    if new_edges == 'random':
      self.add_random_edges()
    elif new_edges == 'random_walk':
      self.add_rw_edges()
    elif new_edges == 'k_hop_lap':
      pass
    elif new_edges == 'k_hop_att':
      self.add_khop_edges(k=2)

  def threshold_edges(self, x, threshold=None):
    # get mean attention
    attention_weights = self.get_attention_weights(x)
    mean_att = attention_weights.mean(dim=1, keepdim=False)
    att_samp_pct = 0.95
    threshold = torch.quantile(mean_att, 1 - att_samp_pct)
    mask = mean_att > threshold
    self.odefunc.edge_index = self.data_edge_index[:, mask.T]
    sampled_attention_weights = self.renormalise_attention(mean_att[mask])
    # print('retaining {} of {} edges'.format(self.odefunc.edge_index.shape[1], self.data_edge_index.shape[1]))
    self.data_edge_index = self.data_edge_index[:, mask.T]
    self.odefunc.edge_weight = sampled_attention_weights #rewiring structure so need to replace any preproc ew's with new ew's
    self.odefunc.attention_weights = sampled_attention_weights

  def forward(self, x):
    t = self.t.type_as(x)
    # self.training = 0
    if self.training:
      # print('Still in training........')
      with torch.no_grad():
        #calc attentions for transition matrix
        attention_weights = self.get_attention_weights(x)
        self.odefunc.attention_weights = attention_weights.mean(dim=1, keepdim=False)

        # Densify and threshold attention weights
        pre_count = self.odefunc.edge_index.shape[1]
        self.densify_edges()
        post_count = self.odefunc.edge_index.shape[1]
        pc_change = post_count /pre_count - 1
        # threshold = torch.quantile(self.odefunc.edge_weight, 1/(pc_change - self.rw_addD))

        self.threshold_edges(x)

    self.odefunc.edge_index = self.data_edge_index
    attention_weights = self.get_attention_weights(x)
    mean_att = attention_weights.mean(dim=1, keepdim=False)
    self.odefunc.edge_weight = mean_att
    self.odefunc.attention_weights = mean_att

    integrator = self.train_integrator if self.training else self.test_integrator
    func = self.odefunc

    if self.adjoint:
        state_dt = integrator(func, x, t, method='dopri5',
                                options={'step_size': 1},
                                adjoint_method='dopri5',
                                adjoint_options={'step_size': 1},
                                atol=self.atol,
                                rtol=self.rtol,
                                adjoint_atol=self.atol_adjoint,
                                adjoint_rtol=self.rtol_adjoint)
    else:
        #state_dt = integrator(func, x, t, method='dopri5')
        state_dt = integrator(func, x, t, method='implicit_adams', options=dict(step_size=2))

    z = state_dt[1]
    return z#, mean_att

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
           
class MEANCURV(nn.Module):
    r"""

    Description
    -----------
    Graph Convolutional Networks (`GCN <https://arxiv.org/abs/1609.02907>`__)

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    n_layers : int
        Number of layers.
    layer_norm : bool, optional
        Whether to use layer normalization. Default: ``False``.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``torch.nn.functional.relu``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    adj_norm_func : func of utils.normalize, optional
        Function that normalizes adjacency matrix. Default: ``GCNAdjNorm``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 residual=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 dropout=0.0):
        super(MEANCURV, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        #self.odeblk = ODEBlock_feat(ODEfunc_feat(out_features))
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       residual=residual if i != n_layers - 1 else False,
                                       dropout=dropout if i != n_layers - 1 else 0.0))
        self.reset_parameters()

    @property
    def model_type(self):
        """Indicate type of implementation."""
        return "torch"

    @property
    def model_name(self):
        return "gcn"

    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of model (logits without activation).

        """
        # att = []
        # edge_idx = []
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = layer(x, adj)
                # x, mean_att, ei = layer(x, adj)
                # att.append(mean_att)
                # edge_idx.append(ei)
        return x#, att, edge_idx



class GCNConv(nn.Module):
    r"""

    Description
    -----------
    GCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 activation=None,
                 residual=False,
                 dropout=0.0):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.time_tensor = torch.tensor([0, 1])
        #self.odeblk = ODEBlock_feat(ODEfunc_feat(out_features))
        
        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None
        self.activation = activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of layer.

        """
        x = self.linear(x)
        
        #block = AttODEblock(ODEFuncTransformerAtt, self.out_features, self.out_features, adj, self.time_tensor)
        block = RewireAttODEblock(ODEFuncTransformerAtt, self.out_features, self.out_features, adj, self.time_tensor, self.training)
        
        block.set_x0(x)
        
        x = block(x)
        
        #x = torch.sparse.mm(adj, x)
        if self.activation is not None:
            x = self.activation(x)
        if self.residual is not None:
            x = x + self.residual(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x#, mean_att, block.odefunc.edge_index
