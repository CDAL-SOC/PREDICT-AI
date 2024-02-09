# !/usr/bin/env python

import torch,pickle,sys,random,os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.types import Device
from scipy import stats
from numpy import argmax
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,accuracy_score,precision_score,recall_score,roc_curve


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def set_device(args, device: Device=None) -> torch.device:
	"""Resolve a :class:`torch.device` given a desired device name."""
	# print(torch.cuda.device_count())
	if device is None or device == "gpu":
		device = "cuda:"+str(args.gpu)
	if isinstance(device, str):
		device = torch.device(device)
	if not torch.cuda.is_available() and device.type == "cuda":
		device = torch.device("cpu")
		print("***No cuda devices were available. CPU will be used.***")
	# device = torch.device("cpu")
	return device

def seed_worker(worker_id):
	worker_seed = torch.initial_seed()
	numpy.random.seed(worker_seed)
	random.seed(worker_seed)

def nan2inf(X):
	return torch.where(torch.isnan(X), torch.zeros_like(X) + np.inf, X)

def save(embs, mapfunc, selected):
	n_dim = embs.shape[1]
	emb_dict = dict()
	for i,emb in enumerate(embs):
		gene = mapfunc[i]
		if gene in selected:
			emb_dict[gene] = emb

	notsaved = list(set(selected).difference(set([k for k in emb_dict.keys()])))
	for gene in notsaved:
		emb_dict[gene] = np.zeros((n_dim))
	filename = 'results/embedding_g2g'
	with open(filename+'.pickle', 'wb') as handle:
		pickle.dump(emb_dict, handle)

	# with open(filename+'.pickle', 'rb') as handle:
	# 	data = pickle.load(handle)
	# print(data)

def convolution(adj:sp.csr_matrix):
	"""adjacency matrix normalization"""
	adj = sp.coo_matrix(adj)
	adj_ = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	# return utils.sparse_to_tuple(adj_normalized)
	return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse_coo_tensor(indices, values, shape)

def evaluate(data, y_preds): #126
	print("\n*** Evaluating results ***")
	y_true = data.test_tcga_dataset.tcga_response
	y_pred = data.test_tcga_dataset.tcga_response.copy()
	y_pred["response"] = y_preds

	y_save = data.test_tcga_dataset.tcga_response.copy()
	y_save["pred"] = y_preds
	idx = random.randint(0, 99)
	y_save.to_csv('results/pred_'+str(idx)+'.csv', index=False)

	drugs_with_enough_support = ["CISPLATIN", "PACLITAXEL", "5-FLUOROURACIL"]
	# drugs_with_enough_support = ['5-FLUOROURACIL','CISPLATIN','CYCLOPHOSPHAMIDE','DOCETAXEL','GEMCITABINE','PACLITAXEL']
	for drug_name in drugs_with_enough_support:
		roc = roc_auc_score(y_true[y_true.drug_name==drug_name].response.values,y_pred[y_pred.drug_name==drug_name].response.values,average="micro",)
		aupr = average_precision_score(y_true[y_true.drug_name==drug_name].response.values,y_pred[y_pred.drug_name==drug_name].response.values,average="micro",)
		print("------ {:s} ------".format(drug_name))
		print("AUROC:{:.4f}, AUPRC:{:.4f}.".format(roc, aupr))

	print("------ All 3 Drugs ------")
	roc = roc_auc_score(y_true.response.values,y_pred.response.values,average="micro",)
	aupr = average_precision_score(y_true.response.values,y_pred.response.values,average="micro",)
	print("AUROC:{:.4f}, AUPRC:{:.4f}.".format(roc,aupr))

def AUROC_AUPRC(y_true, y_pred):
	auroc = roc_auc_score(y_true, y_pred, average="micro",)
	auprc = average_precision_score(y_true, y_pred, average="micro",)
	# print("AUROC:{:.4f}, AUPRC:{:.4f}.".format(auroc,auprc))
	return auroc,auprc


from torch import nn
class BCEFocalLoss(torch.nn.Module):
	def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.reduction = reduction

	def forward(self, _input, target):
		pt = torch.sigmoid(_input)
		# pt = _input
		alpha = self.alpha
		loss = -alpha*(1-pt)**self.gamma*target*torch.log(pt) - (1-alpha)*pt**self.gamma*(1-target)*torch.log(1-pt)
		if self.reduction == 'mean':
			loss = torch.mean(loss)
		elif self.reduction == 'sum':
			loss = torch.sum(loss)
		return loss


def mtlr_neg_log_likelihood(logits:torch.Tensor,target:torch.Tensor,model:torch.nn.Module,C1:float,average:bool=False):
	def masked_logsumexp(x:torch.Tensor,mask:torch.Tensor,dim:int=-1):
		max_val,_ = (x*mask).max(dim=dim)
		max_val = torch.clamp_min(max_val,0)
		return torch.log(torch.sum(torch.exp(x-max_val.unsqueeze(dim))*mask,dim=dim))+max_val

	censored = target.sum(dim=1) > 1
	nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
	nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

	# the normalising constant
	norm = torch.logsumexp(logits, dim=1).sum()
	nll_total = -(nll_censored + nll_uncensored - norm)
	if average:
		nll_total = nll_total / target.size(0)

	# # L2 regularization
	# for k, v in model.named_parameters():
	# 	if "mtlr_weight" in k:
	# 		nll_total += C1/2 * torch.sum(v**2)

	return nll_total