# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import linalg
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import *
from utils import *
from torch.autograd import Variable
import sys


# [ablation] main
class cola_gnn(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.x_h = 1  # input feature dimension- our data is 1D array //input size for rnn
        self.f_h = data.m  # output dimension
        self.m = data.m  # number of locations
        self.d = data.d  # todo:should find out
        self.w = args.window
        self.h = args.horizon
        self.adj = data.adj
        self.o_adj = data.orig_adj
        self.static = normalize_static(data.static)  # shape (5,50)
        self.static = self.static.transpose()  # shape (50,5)
        if args.cuda:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden  # input dimension for final output layer
        half_hid = int(self.n_hidden / 2)  # no of hidden units- half from GNN and half from RNN
        self.V = Parameter(torch.Tensor(half_hid))  # weight matrix of the attention mechanism, with shape (half_hid,)
        self.bv = Parameter(torch.Tensor(1))  # bias term of the attention mechanism, with shape (1,)
        self.W1 = Parameter(torch.Tensor(half_hid,
                                         self.n_hidden))  # weight matrix of the first feedforward layer, with shape (half_hid, self.n_hidden)
        self.b1 = Parameter(torch.Tensor(half_hid))  # bias term of the first feedforward layer, with shape (half_hid,)
        self.W2 = Parameter(torch.Tensor(half_hid,
                                         self.n_hidden))  # weight matrix of the second feedforward layer, with shape (half_hid, self.n_hidden)
        self.act = F.elu  # activation function applied to the output of the feedforward layers, which is F.elu in this case
        self.Wb = Parameter(
            torch.Tensor(self.m, self.m))  # weight matrix of the graph convolution layer, with shape (self.m, self.m)
        self.wb = Parameter(torch.Tensor(1))  # bias term of the graph convolution layer, with shape (1,)
        self.k = args.k  ##kernel size of the convolutional layers
        self.conv = nn.Conv1d(1, self.k, self.w)  # todo why using kernel size two times?
        long_kernal = self.w // 2  ##half the size of window w
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal,
                                   dilation=2)  # 1D convolutional layer used to extract long-range dependencies from the input sequence
        long_out = self.w - 2 * (
                    long_kernal - 1)  # output size of self.conv_long, which is used to concatenate the local and long-range features
        self.n_spatial = 10  # number of output features of the final graph convolution layer
        self.conv1 = GraphConvLayer((1 + long_out) * self.k,
                                    self.n_hidden )  # first graph convolutional layer, which takes as input the concatenation of local and long-range features and applies a graph convolution
        self.conv2 = GraphConvLayer(self.n_hidden ,
                                    self.n_spatial)  # second graph convolutional layer, which takes as input the output of self.conv1

        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer,
                               dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer,
                              dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN(input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer,
                              dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        hidden_size = (
                                  int(args.bi) + 1) * self.n_hidden  # Calculate the hidden size of the RNN layer. If the RNN is bidirectional, the hidden size is twice the n_hidden size, otherwise it is equal to n_hidden.
        self.out = nn.Linear(hidden_size + self.n_spatial, 1)

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)  # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):
        '''
        Args:  x: (batch, time_step, m)
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        '''
        b, w, m = x.size()
        orig_x = x
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1)
        '''permute function, which rearranges the dimensions of the tensor in the order specified in the arguments.
        The contiguous function ensures that the memory of the tensor is laid out in a contiguous block, which is a requirement for certain operations.'''
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:, -1, :]  # Retrieve the last hidden state of the RNN output for each sequence in the batch.
        last_hid = last_hid.view(-1, self.m,
                                 self.n_hidden)  # Reshape the tensor to be of shape (batch_size, m, n_hidden).
        out_temporal = last_hid  # [b, m, 20]
        hid_rpt_m = last_hid.repeat(1, self.m, 1).view(b, self.m, self.m,
                                                       self.n_hidden)  # b,m,m,w continuous m Repeat each tensor in the batch m times in the second dimension
        hid_rpt_w = last_hid.repeat(1, 1, self.m).view(b, self.m, self.m,
                                                       self.n_hidden)  # b,m,m,w continuous w one window data
        #learning attention matrix
        a_mx = self.act(
            hid_rpt_m @ self.W1.t() + hid_rpt_w @ self.W2.t() + self.b1) @ self.V + self.bv  # row, all states influence one state
        ''' Perform matrix multiplication and addition operations to compute the influence of the all previous states on each current state, followed by activation function, and normalize it.'''
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)

        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:, :, i:i + 1].permute(0, 2, 1).contiguous()
            r = self.conv(h_tmp)  # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l, dim=1)
        r_long_l = torch.stack(r_long_l, dim=1)
        r_l = torch.cat((r_l, r_long_l), -1)
        r_l = r_l.view(r_l.size(0), r_l.size(1), -1)
        r_l = torch.relu(r_l)

        #geo-ad matrix
        adjs = self.adj.repeat(b, 1)
        adjs = adjs.view(b, self.m, self.m)

        #combine learning attention and geo-ad matrix M
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        #a_mx = adjs * c + a_mx * (1 - c)
        adj = a_mx
        #adj=adjs

        x = r_l
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))

        out = torch.cat((out_spatial, out_temporal), dim=-1)
        out = self.out(out)
        out = torch.squeeze(out)

        #l_g = (a_mx - adjs)
        l_g = (a_mx - adjs)
        l_g = torch.linalg.norm(l_g, ord=2,dim=(1,2), keepdim=False)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :];  # Step backward # [batch, res_window, m]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window);  # [batch*m, res_window]
            z = self.residual(z);  # [batch*m, 1]
            z = z.view(-1, self.m);  # [batch, m]
            out = out * self.ratio + z;  # [batch, m]

        return  l_g, out, None
