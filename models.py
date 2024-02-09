r"""An implementation of the Unsup model."""
import torch,logging,time,sys,pickle
from torch import nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import utils as utils
from utils import BCEFocalLoss
from sklearn.model_selection import train_test_split
from torchmtlr.utils import (make_time_bins,encode_survival,make_optimizer,reset_parameters,)
from torch.utils.data import TensorDataset,DataLoader,Sampler
from torchmtlr import MTLR
from torchmtlr import (mtlr_neg_log_likelihood,mtlr_survival,mtlr_risk,)
from lifelines.utils import concordance_index
import shap


class DrugResponse:
	def __init__(self, data, args, device=None):
		print("*** Running DrugResponse model. ***")
		self.data,self.args = data,args
		self.device = device
		# self.device = utils.set_device(args, device)
		# print(self.device)

		if args.model == 'TransformerMTLR':
			self.tMTLR = TransformerMTLR(self.args,data.n_vocab,data.n_drugs,self.device,embedding_dim=17,d_model=self.args.hid_dim,num_layers=8,num_time_bins=9)
			self.tMTLR.to(self.device)
			print(self.tMTLR)
			print("# Params:", sum(p.numel() for p in self.tMTLR.parameters() if p.requires_grad))
			self.tformer()
		elif args.model == 'TransformerDRP':
			self.tDRP = TransformerDRP(self.args,data.n_vocab,data.n_drugs,data.gene2id, self.device)
			self.tDRP.to(self.device)
			print("# Params:", sum(p.numel() for p in self.tDRP.parameters() if p.requires_grad))
			self.tpred()


	def tformer(self): #self.data.annodata | self.data.annovar_val: torch.Size([12176, 17])
		anno = self.data.annodata
		input_ids,attention,annovar_ids,times,events,drugs = anno['input_ids'],anno['attention_mask'],anno['annovar_mask'],anno['times'],anno['events'],anno['drugs']
		time_bins = make_time_bins(times=times, event=events, num_bins=9)
		targets = encode_survival(times, events, time_bins)
		dataloader = DataLoader(TensorDataset(input_ids.to(self.device),attention.to(self.device),annovar_ids.to(self.device),
			times.to(self.device),events.to(self.device),drugs.to(self.device),targets.to(self.device)),batch_size=self.args.batchsize,shuffle=False,)
		annovar = self.data.annovar_val.to(self.device,torch.float32) #torch.Size([12176, 17])
		drugsfp = self.data.drugs_fpval.to(self.device,torch.float32)

		optimizer = make_optimizer(torch.optim.Adam, self.tMTLR, lr=self.args.lr, weight_decay=self.args.weight_decay)
		reset_parameters(self.tMTLR)

		losses,CIindex = [],[]
		for epoch in range(self.args.epochs):
			if (epoch+1)%20 == 0 or epoch+1 == 1:
				self.tMTLR.train()
				events_pred,times_pred,risk_pred = [],[],[]
				for input_ids_i,attention_i,annovar_ids_i,times_i,events_i,drug_i,yi in dataloader:
					optimizer.zero_grad()
					y_pred = self.tMTLR(input_ids_i, attention_i, annovar_ids_i, annovar, drug_i, drugsfp) #torch.Size([512,10])
					loss = mtlr_neg_log_likelihood(y_pred, yi, self.tMTLR, C1=1.0, average=True)
					loss.backward()
					optimizer.step()

					risk = mtlr_risk(y_pred).tolist()
					events_pred += events_i.tolist()
					times_pred += times_i.tolist()
					risk_pred += risk
				CI_pred = concordance_index(np.array(times_pred),-np.array(risk_pred),np.array(events_pred),)
				print("*** Epoch [{}/{}], loss:{:.4f}, ci_index:{:.4f}.".format(epoch+1,self.args.epochs,loss.item(),CI_pred.item()))
				losses.append(loss.item())
				CIindex.append(CI_pred.item())
			else:
				self.tMTLR.train()
				for input_ids_i,attention_i,annovar_ids_i,times_i,events_i,drug_i,yi in dataloader:
					optimizer.zero_grad()
					y_pred = self.tMTLR(input_ids_i, attention_i, annovar_ids_i, annovar, drug_i, drugsfp) #torch.Size([512,10])
					loss = mtlr_neg_log_likelihood(y_pred, yi, self.tMTLR, C1=1.0, average=True)
					loss.backward()
					optimizer.step()

		torch.save(self.tMTLR.embs_gene.state_dict(),f"../data/model_checkpoints/GENE_EMBEDDING_from_survival_prediction_model_CRCCLC.pt",)
		torch.save(self.tMTLR.embs_drug.state_dict(),f"../data/model_checkpoints/DRUG_EMBEDDING_from_survival_prediction_model_CRCCLC.pt",)
		torch.save(self.tMTLR.transformer_encoder.state_dict(),f"../data/model_checkpoints/Pretrained_transformer_encoder_model_CRCCLC.pt",)
		np.save('results/TransformerMTLR_losses.npy', np.array(losses, dtype=object), allow_pickle=True)
		np.save('results/TransformerMTLR_CIindex.npy', np.array(CIindex, dtype=object), allow_pickle=True)

	def tpred(self):
		print('*** Transformer-based Drug Response Prediction ***')
		if self.args.opt == 'Adam':
			optimizer = optim.Adam(self.tDRP.parameters(), lr=self.args.lr)
		elif self.args.opt == 'SGD':
			optimizer = optim.SGD(self.tDRP.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
		elif self.args.opti == 'AdaGrad':
			optimizer = optim.Adagrad(self.tDRP.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

		criteria_recist = BCEFocalLoss()
		criteria_audrc = nn.MSELoss()

		annovar = self.data.annovar_val.to(self.device,torch.float32) #torch.Size([12176, 17])
		annovar_ccle = self.data.anno_ccle.to(self.device,torch.float32)
		annovar_tcga = self.data.anno_tcga.to(self.device,torch.float32)
		drugsfp = self.data.drugs_fpval.to(self.device,torch.float32)

		# Training
		best,cnt_wait = 1e9,0
		best_auroc,best_auprc,best_epoch = 0,0,0
		# self.tDRP.train()
		for epoch in range(self.args.epochs):
			self.tDRP.train()
			loss_recist = 0.0
			y_true,y_preds = [],[]
			for mat, anno, mask, drug, y in self.data.train_tcga: #[256,198],[256,378],[256,576],[256,2048]
				mat,anno,mask,drug = mat.to(self.device,torch.int32),anno.to(self.device,torch.int32),torch.Tensor(mask).to(self.device),drug.to(self.device,torch.float32)
				recist_pred = self.tDRP.patient_predictor(mat, anno, mask, drug, annovar_tcga)
				loss = criteria_recist(recist_pred,y.unsqueeze(1).to(self.device,torch.float32))
				y_preds.extend(list(recist_pred.flatten().detach().cpu().numpy()))
				y_true.extend(list(y.detach().cpu().numpy()))
				loss_recist += loss
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			loss_recist = loss_recist/len(self.data.train_tcga)
			auroc,auprc = utils.AUROC_AUPRC(y_true, y_preds)
			# print("--AUROC:{:.4f}, AUPRC:{:.4f}.".format(auroc,auprc))

			loss_audrc = 0.0
			for mat, anno, mask, drug, y in self.data.train_cl:
				mat,anno,mask,drug = mat.to(self.device,torch.int32),anno.to(self.device,torch.int32),torch.Tensor(mask).to(self.device),drug.to(self.device,torch.float32)
				audrc_pred = self.tDRP.cellline_predictor(mat, anno, mask, drug, annovar_ccle) #torch.Size([256, 1])
				loss = criteria_audrc(audrc_pred,y.unsqueeze(1).to(self.device,torch.float32))
				loss_audrc += loss
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			loss_audrc = loss_audrc/len(self.data.train_cl)

			loss_survival = 0.0
			for input_ids_i,attention_i,annovar_ids_i,times_i,events_i,drug_i,yi in self.data.train_nsclc:
				y_pred = self.tDRP.survival_predictor(input_ids_i,attention_i,annovar_ids_i,annovar,drug_i,drugsfp) #torch.Size([512,10])
				loss = mtlr_neg_log_likelihood(y_pred, yi, self.tDRP, C1=1.0, average=True)
				optimizer.zero_grad()
				loss_survival += loss
				loss.backward()
				optimizer.step()
			loss_survival = loss_survival/len(self.data.train_nsclc)

			total_loss = loss_recist + loss_audrc + loss_survival
			all_rel_losses = {'Total':[total_loss],'RECIST_prediction':[loss_recist],'Survival_prediction':[loss_survival]}

			self.tDRP.eval()
			y_true,y_preds = [],[]
			for mat, anno, mask, drug, y in self.data.val_tcga:
				mat,anno,mask,drug = mat.to(self.device,torch.int32),anno.to(self.device,torch.int32),torch.Tensor(mask).to(self.device),drug.to(self.device,torch.float32)
				recist_pred = self.tDRP.patient_predictor(mat, anno, mask, drug, annovar_tcga)
				y_preds.extend(list(recist_pred.flatten().detach().cpu().numpy()))
				y_true.extend(list(y.detach().cpu().numpy()))
			val_auroc,val_auprc = utils.AUROC_AUPRC(y_true, y_preds)

			self.tDRP.eval()
			y_true,y_preds = [],[]
			for mat, anno, mask, drug, y in self.data.test_tcga:
				mat,anno,mask,drug = mat.to(self.device,torch.int32),anno.to(self.device,torch.int32),torch.Tensor(mask).to(self.device),drug.to(self.device,torch.float32)
				recist_pred = self.tDRP.patient_predictor(mat, anno, mask, drug, annovar_tcga)
				y_preds.extend(list(recist_pred.flatten().detach().cpu().numpy()))
				y_true.extend(list(y.detach().cpu().numpy()))
			test_auroc,test_auprc = utils.AUROC_AUPRC(y_true, y_preds)
			print("Train-AUROC:{:.4f}, AUPRC:{:.4f}; Val-AUROC:{:.4f}, AUPRC:{:.4f}; Test-AUROC:{:.4f}, AUPRC:{:.4f}.".format(auroc,auprc,val_auroc,val_auprc,test_auroc,test_auprc))

			# if val_auroc >= best_auroc or val_auprc >= best_auprc:
			if val_auprc >= best_auprc and epoch >= 0:
				torch.save(self.tDRP.state_dict(), 'tmp/best_tDRP_model'+str(self.args.gpu)+'.pkl')
				best_auroc,best_auprc = val_auroc,val_auprc
				best_epoch = epoch
				print("*** Train-AUROC:{:.4f}, AUPRC:{:.4f}; Val-AUROC:{:.4f}, AUPRC:{:.4f}; Test-AUROC:{:.4f}, AUPRC:{:.4f}.".format(auroc,auprc,val_auroc,val_auprc,test_auroc,test_auprc))
				# print("*** Train-AUROC:{:.4f}, AUPRC:{:.4f}; Val-AUROC:{:.4f}, AUPRC:{:.4f}.".format(auroc,auprc,val_auroc,val_auprc))

			torch.save(self.tDRP.state_dict(), 'tmp/last_tDRP_model'+str(self.args.gpu)+'.pkl')

			if (epoch+1)%20 == 0 or epoch+1 == 1:
				# print(all_rel_losses)
				print("### Epoch [{}/{}], loss:{:.6f}".format(epoch+1, self.args.epochs, total_loss.item()))
			if total_loss < best:
				cnt_wait,best = 0,total_loss
			else:
				cnt_wait += 1
			if cnt_wait == self.args.patience:
				print("Early stopping!")
				break


		self.tDRP.eval()
		self.tDRP.load_state_dict(torch.load('tmp/last_tDRP_model'+str(self.args.gpu)+'.pkl'))
		y_true,y_preds = [],[]
		for mat, anno, mask, drug, y in self.data.test_tcga:
			mat,anno,mask,drug = mat.to(self.device,torch.int32),anno.to(self.device,torch.int32),torch.Tensor(mask).to(self.device),drug.to(self.device,torch.float32)
			recist_pred = self.tDRP.patient_predictor(mat, anno, mask, drug, annovar_tcga)
			y_preds.extend(list(recist_pred.flatten().detach().cpu().numpy()))
			y_true.extend(list(y.detach().cpu().numpy()))
		test_auroc,test_auprc = utils.AUROC_AUPRC(y_true, y_preds)
		print("Test-AUROC:{:.4f}, AUPRC:{:.4f}.".format(test_auroc,test_auprc))
		# ### Evaluation
		utils.evaluate(self.data, y_preds)
		print()

		self.tDRP.eval()
		print('Loading {}-th epoch.'.format(best_epoch))
		self.tDRP.load_state_dict(torch.load('tmp/best_tDRP_model'+str(self.args.gpu)+'.pkl'))
		y_true,y_preds = [],[]
		for mat, anno, mask, drug, y in self.data.test_tcga:
			mat,anno,mask,drug = mat.to(self.device,torch.int32),anno.to(self.device,torch.int32),torch.Tensor(mask).to(self.device),drug.to(self.device,torch.float32)
			recist_pred = self.tDRP.patient_predictor(mat, anno, mask, drug, annovar_tcga)
			y_preds.extend(list(recist_pred.flatten().detach().cpu().numpy()))
			y_true.extend(list(y.detach().cpu().numpy()))
		test_auroc,test_auprc = utils.AUROC_AUPRC(y_true, y_preds)
		print("Test-AUROC:{:.4f}, AUPRC:{:.4f}.".format(test_auroc,test_auprc))

		# print(y_preds)
		# ### Evaluation
		utils.evaluate(self.data, y_preds)
		

# ### Transformer-based Survival Prediction Model
class TransformerMTLR(nn.Module):
	def __init__(self,args,n_vocab,n_drugs,device, embedding_dim=23,d_model=16,nhead=4,dim_feedforward=64,num_layers=4,num_time_bins=9,) -> None:
		super(TransformerMTLR, self).__init__()
		self.args = args
		self.device = device
		self.embs_gene = nn.Embedding(num_embeddings=n_vocab, embedding_dim=64, padding_idx=1)
		# self.embs_drug = nn.Embedding(num_embeddings=n_drugs, embedding_dim=64, padding_idx=1)
		# self.embs_drug = nn.Sequential(nn.Linear(2048, 128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128, 64))
		# self.embs_drug = nn.Linear(2048, 64)
		self.embs_drug = self.drugNN(2048, 1024, 512, 64)
		self.fc = nn.Linear(in_features=23, out_features=64)
		self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64,nhead=nhead,dropout=self.args.dropout,batch_first=True,),num_layers=num_layers,)
		# self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=32,nhead=8,dropout=0.3,batch_first=True,),num_layers=4,)
		self.mtlr = MTLR(in_features=64*2, num_time_bins=num_time_bins)

	def drugNN(self, In, hidden1, hidden2, out):
		return nn.Sequential(nn.Linear(In,hidden1),nn.BatchNorm1d(hidden1),nn.ReLU(),
			nn.Linear(hidden1,hidden2),nn.BatchNorm1d(hidden2),nn.ReLU(),nn.Linear(hidden2,out))

	def forward(self, input_ids, attention, annovar_ids, annovar, drugs, drugsfp):
		x_input,x_attention,x_annovar = (self.embs_gene(input_ids),attention==0,annovar[annovar_ids],)
		x_drug = self.embs_drug(drugsfp[drugs])
		# x_drug = self.embs_drug(drugs)
		x_gene = x_input*self.fc(x_annovar)
		x_gene = self.transformer_encoder(x_gene, src_key_padding_mask=x_attention)
		x_gene = x_gene.mean(dim=1)
		x_cat = torch.cat((x_gene, x_drug), dim=1)
		x = self.mtlr(x_cat)
		return x


# ### Transformer-based Drug Response Prediction Model
class TransformerDRP(nn.Module):# Used for training 2 tasks: cellline-drug AUDRC prediction(regression) and patient-drug RECIST prediction(classification)
	def __init__(self, args, ipt_dim, n_drug, gene2id, device, single=False):
		super(TransformerDRP, self).__init__()
		self.args = args
		self.device = device
		self.gene2id = gene2id
		# pretrained gene embedding from survival prediction model
		self.geneEMB_survival = nn.Embedding(num_embeddings=ipt_dim, embedding_dim=64, padding_idx=1)
		# self.drugEMB_survival = nn.Embedding(num_embeddings=n_drug, embedding_dim=64, padding_idx=1)
		self.fc_annovar = nn.Linear(in_features=23, out_features=64)
		self.drug_embedder = self.drugNN(2048, 1024, 256, 64)
		# self.drug_embedder = nn.Linear(2048, 64)
		self.recist_predictor = nn.Sequential(self.predNN(128, 32, 16, 1), ) # takes as input concatenated representation of patient and drug
		self.audrc_predictor = nn.Sequential(self.predNN(128, 32, 16, 1), ) # takes as input concatenated representation of cell line and drug
		# self.survival_predictor = nn.Sequential(self.predNN(128, 32, 16, 1), )
		self.embedder = nn.Sequential(nn.Linear(281, 128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128, 64))
		self.survival_embedder = nn.Sequential(nn.Linear(44*64, 128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128, 64))
		self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64,nhead=8,dropout=self.args.dropout,batch_first=True,),num_layers=8,)
		self.mtlr = MTLR(in_features=128, num_time_bins=9)

		self.load_pretraining()

	def load_pretraining(self):
		self.geneEMB_survival.load_state_dict(torch.load(f"../data/model_checkpoints/GENE_EMBEDDING_from_survival_prediction_model_CRCCLC.pt",map_location=str(self.device),))
		# self.drug_embedder.load_state_dict(torch.load(f"../data/model_checkpoints/DRUG_EMBEDDING_from_survival_prediction_model_CRCCLC.pt",map_location=str(self.device),))
		self.transformer_encoder.load_state_dict(torch.load(f"../data/model_checkpoints/Pretrained_transformer_encoder_model_CRCCLC.pt",map_location=str(self.device),))

	def drugNN(self, In, hidden1, hidden2, out):
		return nn.Sequential(nn.Linear(In,hidden1),nn.BatchNorm1d(hidden1),nn.ReLU(),
			nn.Linear(hidden1,hidden2),nn.BatchNorm1d(hidden2),nn.ReLU(),nn.Linear(hidden2,out))
	def predNN(self, In, hidden1, hidden2, out):
		return nn.Sequential(nn.Linear(In,hidden1),nn.BatchNorm1d(hidden1),nn.ReLU(),
			nn.Linear(hidden1,hidden2),nn.BatchNorm1d(hidden2),nn.ReLU(),nn.Linear(hidden2,out),nn.Sigmoid())

	def cellline_predictor(self, cl_mut_input,cl_anno_input,cl_mut_input_mask,cl_drug_input,annovar):
		cl_drug_emb = self.drug_embedder(cl_drug_input)
		cl_mut_emb = self.geneEMB_survival(cl_mut_input)
		cl_anno_emb = self.fc_annovar(annovar[cl_anno_input]) #torch.Size([256, 378, 64])
		cl_mut_emb = torch.cat((cl_mut_emb, cl_anno_emb), dim=1) #torch.Size([256, 576, 64])
		cl_mut_emb = self.transformer_encoder(cl_mut_emb,src_key_padding_mask=cl_mut_input_mask)
		cl_mut_emb = cl_mut_emb.mean(dim=2)
		# cl_mut_emb = torch.flatten(cl_mut_emb, start_dim=1)
		cl_mut_emb = self.embedder(cl_mut_emb)
		cl_drug_cat_emb = torch.cat((cl_mut_emb, cl_drug_emb), dim=1)
		audrc_prediction = self.audrc_predictor(cl_drug_cat_emb)
		return audrc_prediction

	def patient_predictor(self, patient_mut_input,patient_anno_input,patient_mut_input_mask,patient_drug_input,annovar):
		patient_drug_emb = self.drug_embedder(patient_drug_input) #torch.Size([256, 64])
		patient_mut_emb = self.geneEMB_survival(patient_mut_input) #torch.Size([256, 198, 64])
		patient_anno_emb = self.fc_annovar(annovar[patient_anno_input]) #torch.Size([256, 378, 64])
		patient_mut_emb = torch.cat((patient_mut_emb, patient_anno_emb), dim=1) #torch.Size([256, 576, 64])
		patient_mut_emb = self.transformer_encoder(patient_mut_emb,src_key_padding_mask=patient_mut_input_mask)
		patient_mut_emb = patient_mut_emb.mean(dim=2)
		# patient_mut_emb = torch.flatten(patient_mut_emb, start_dim=1)
		patient_mut_emb = self.embedder(patient_mut_emb)
		patient_drug_cat_emb = torch.cat((patient_mut_emb, patient_drug_emb), dim=1)
		recist_prediction = self.recist_predictor(patient_drug_cat_emb)
		return recist_prediction

	def survival_predictor(self, input_ids, attention, annovar_ids, annovar, drugs, drugsfp):
		x_input,x_attention,x_annovar = (self.geneEMB_survival(input_ids),attention==0,annovar[annovar_ids],)
		x_drug = self.drug_embedder(drugsfp[drugs])
		# x_drug = self.embs_drug(drugs)
		x_gene = x_input*self.fc_annovar(x_annovar)
		x_gene = self.transformer_encoder(x_gene, src_key_padding_mask=x_attention)
		x_gene = x_gene.mean(dim=1)
		x_cat = torch.cat((x_gene, x_drug), dim=1)
		x = self.mtlr(x_cat)
		return x

	def mtlr_neg_log_likelihood(self, logits:torch.Tensor,target:torch.Tensor,model:torch.nn.Module,C1:float,average:bool=False):
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
		# if average:
		# 	nll_total = nll_total / target.size(0)
		# # L2 regularization
		# for k, v in model.named_parameters():
		# 	if "mtlr_weight" in k:
		# 		nll_total += C1/2 * torch.sum(v**2)
		return nll_total
