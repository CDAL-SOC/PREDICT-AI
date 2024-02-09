# !/usr/bin/env python
# -*- coding: utf8 -*-

import gzip,sys,torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset,DataLoader,TensorDataset
from functools import cached_property
from torchtext.vocab import build_vocab_from_iterator
from collections import defaultdict
from torchmtlr.utils import make_time_bins,encode_survival
from typing import Optional,Union
from transformers import BatchEncoding,TensorType
from sklearn.model_selection import train_test_split
from utils import seed_worker


class MedicalDATA(object):
	"""docstring for MedicalDATA"""
	def __init__(self, args, device):
		super(MedicalDATA, self).__init__()
		self.args = args
		self.device = device

		self.g = torch.Generator()
		self.g.manual_seed(self.args.seed)

		### Loading Drugs
		drug_names = np.array(['5-FLUOROURACIL','CYCLOPHOSPHAMIDE','DOCETAXEL','GEMCITABINE','CISPLATIN','PACLITAXEL'])
		drug_names_to_idx_map = dict(zip(drug_names,range(len(drug_names))))
		drug_fp = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv",index_col=0) #[408 rows x 2048 columns]
		drug_names_fp = drug_fp.index.to_list() #408

		### Loading annotation
		samples,gene324,annovar = self.load_annotation()
		# print(samples)

		self.gene324 = gene324
		self.annovar_val = torch.tensor(np.concatenate((np.ones((1,len(annovar.columns[2:-1])),dtype=np.int64),annovar[annovar.columns[2:-1]].values,),axis=0,))
		# print(self.annovar_val.shape)
		self.annovar_dict = {"@".join(k.split(" ")):(v+1) for v, k in annovar.input.to_dict().items()} #12175
		# print(self.annovar_val)

		drug_samples = list(set(samples['drug_name'].to_list()))
		drugs_notin_fp = [val for val in drug_samples if val not in drug_names_fp] #['ATEZOLIZUMAB','RAMUCIRUMAB','NIVOLUMAB','PEMBROLIZUMAB','5-Fluorouracil']
		drugs_notin_fp_pd = pd.DataFrame(np.zeros((len(drugs_notin_fp),2048),dtype=int),index=drugs_notin_fp,columns=[str(i) for i in range(2048)])
		self.drugs_fp = pd.concat([drug_fp,drugs_notin_fp_pd],axis=0) #[413 rows x 2048 columns]
		# print(self.drugs_fp)
		self.drugs_fpval = torch.tensor(self.drugs_fp.values) #torch.Size([413, 2048])

		### MAP DRUGS to IDS
		drugs = self.drugs_fp.index.tolist() #413
		self.drug2id = dict(zip(drugs, [i for i in range(len(drugs))]))
		samples['drug_name'] = samples['drug_name'].apply(lambda x: self.drug2id[x]) #[20754 rows x 25 columns]
		self.drugs_name = drugs

		texts = self.get_texts(samples) #Length:2312 #Length:822
		# print(texts)
		token_generator = self.yield_tokens(texts, gene324)
		vocab = build_vocab_from_iterator(token_generator, specials=["<s>","<pad>","</s>","<unk>"])
		tok = Tokenizer(vocab, self.annovar_dict)
		self.annodata = tok(texts, return_tensors="pt", padding=True)
		times = torch.tensor(texts.index.get_level_values("tt_pfs_m_g_mos").values)
		events = torch.tensor(texts.index.get_level_values("pfs_m_g_status").values, dtype=torch.int64)
		self.annodata["times"],self.annodata["events"] = times,events
		drugs = torch.tensor(texts.index.get_level_values("drug_name"))
		self.annodata["drugs"] = drugs

		# sys.exit()

		gene2id = {vocab.lookup_token(idx):idx for idx in range(len(vocab))} #276
		unknown = list(gene324.difference(set([k for k,v in gene2id.items()]))) #55
		idx = len(vocab)
		for gene in unknown:
			gene2id[gene] = idx
			idx += 1
		self.gene2id = gene2id
		self.id2gene = id2gene = {v:k for k,v in gene2id.items()}
		self.n_vocab = len(gene2id) #331
		self.n_drugs = len(self.drug2id)
		self.pretrain_tok = tok

		# Loading mutations
		ccle2gens_mat,ccle2muts_mat,ccle_mask,tcga2gens_mat,tcga2muts_mat,tcga_mask,self.anno_ccle,self.anno_tcga = self._load_mutation_()
		# self.annoval_ccle = torch.tensor(np.concatenate((np.ones((1,len(anno_ccle.columns[1:])),dtype=np.int64),anno_ccle[anno_ccle.columns[1:]].values,),axis=0,))
		# self.annodict_ccle = {"@".join(k.split(" ")):(v+1) for v, k in anno_ccle.mutation.to_dict().items()} #12175
		# self.annoval_tcga = torch.tensor(np.concatenate((np.ones((1,len(anno_tcga.columns[1:])),dtype=np.int64),anno_tcga[anno_tcga.columns[1:]].values,),axis=0,))
		# self.annodict_tcga = {"@".join(k.split(" ")):(v+1) for v, k in anno_tcga.point_mutation.to_dict().items()} #12175
		# print(self.annoval_ccle.shape,self.annoval_tcga.shape)

		# sys.exit()

		# Cell Line
		train_cell_line_dataset = AggCategoricalAnnotatedCellLineDatasetFilteredByDrug(is_train=True,filter_for="tcga",sample_id=args.sample)
		train_cl_features,train_cl_y = self.process_celline(train_cell_line_dataset,ccle2gens_mat,ccle2muts_mat,ccle_mask,drug_fp)
		self.train_cl = DataLoader(CustomDataset_cl(train_cl_features),batch_size=args.batchsize,shuffle=True,worker_init_fn=seed_worker,generator=self.g) #train

		# TCGA
		train_pdx_dataset = AggCategoricalAnnotatedTcgaDatasetFilteredByDrug(is_train=True,filter_for="tcga",sample_id=args.sample)
		train_tcga_features,train_tcga_y = self.process_tcga(train_pdx_dataset,tcga2gens_mat,tcga2muts_mat,tcga_mask,drug_fp)
		self.test_tcga_dataset = AggCategoricalAnnotatedTcgaDatasetFilteredByDrug(is_train=False,filter_for="tcga",sample_id=args.sample)
		test_tcga_features,test_tcga_y = self.process_tcga(self.test_tcga_dataset,tcga2gens_mat,tcga2muts_mat,tcga_mask,drug_fp)

		# self.train_tcga = DataLoader(CustomDataset_tcga(train_tcga_features),batch_size=args.batchsize,shuffle=True,worker_init_fn=seed_worker,generator=self.g) #train
		train_tcga_feats,val_tcga_feats = train_test_split(train_tcga_features,test_size=0.1)
		self.train_tcga = DataLoader(CustomDataset_tcga(train_tcga_feats),batch_size=args.batchsize,shuffle=True,worker_init_fn=seed_worker,generator=self.g) #train
		self.val_tcga = DataLoader(CustomDataset_tcga(val_tcga_feats),batch_size=args.batchsize,shuffle=False,worker_init_fn=seed_worker,generator=self.g) #val
		self.test_tcga = DataLoader(CustomDataset_tcga(test_tcga_features),batch_size=args.batchsize,shuffle=False,worker_init_fn=seed_worker,generator=self.g) #test

		self.train_nsclc = self._load_NSCLC()
		# sys.exit()

	def _load_NSCLC(self):
		samples,annoval,gene324 = self.samples_clc,self.annovar_clc_max,self.gene324
		samples['drug_name'] = samples['drug_name'].apply(lambda x: self.drug2id[x])
		texts = self.get_texts(samples) #Length:2312 #Length:822
		# print(texts)
		annodata = self.pretrain_tok(texts, return_tensors="pt", padding=True)
		times = torch.tensor(texts.index.get_level_values("tt_pfs_m_g_mos").values)
		events = torch.tensor(texts.index.get_level_values("pfs_m_g_status").values, dtype=torch.int64)
		drugs = torch.tensor(texts.index.get_level_values("drug_name"))

		input_ids,attention,annovar_ids = annodata['input_ids'],annodata['attention_mask'],annodata['annovar_mask']
		time_bins = make_time_bins(times=times, event=events, num_bins=9)
		targets = encode_survival(times, events, time_bins)
		dataloader = DataLoader(TensorDataset(input_ids.to(self.device),attention.to(self.device),annovar_ids.to(self.device),times.to(self.device),events.to(self.device),
			drugs.to(self.device),targets.to(self.device)),batch_size=self.args.batchsize,shuffle=False,worker_init_fn=seed_worker,generator=self.g)
		return dataloader

	def _load_mutation_(self,):
		fpath_ccle_mutation = '/data/hansheng/DruID/data/pretrain/clinvar_anno_gpd_features_ccle.csv'
		fpath_tcga_mutation = '/data/hansheng/DruID/data/pretrain/clinvar_anno_gpd_features_tcga.csv'

		clnsigs2cate = {'Pathogenic':'Pathogenic','Pathogenic|drug_response|other':'Pathogenic','Pathogenic/Likely_pathogenic':'Pathogenic',
						'Likely_pathogenic':'Pathogenic','Pathogenic/Likely_pathogenic|other':'Pathogenic','drug_response':'Pathogenic',
						'Likely_pathogenic|other':'Pathogenic','Pathogenic|risk_factor':'Pathogenic','Pathogenic/Likely_pathogenic|drug_response':'Pathogenic',
						'Likely_risk_allele':'Pathogenic','risk_factor':'Pathogenic','Likely_benign':'Benign','Benign/Likely_benign':'Benign','Benign':'Benign',
						'.':'Unknown','Uncertain_significance':'Unknown','Conflicting_interpretations_of_pathogenicity':'Unknown','not_provided':'Unknown',
						'Conflicting_interpretations_of_pathogenicity|other':'Unknown','Uncertain_significance|drug_response':'Unknown','other':'Unknown',
						'Affects':'---','None':'None'}
		### CCLE
		raw_ccle = pd.read_csv(fpath_ccle_mutation)
		# print(raw_ccle)
		ccle2gens,ccle2muts = defaultdict(set),defaultdict(set)
		for idx,row in raw_ccle.iterrows():
			ccle_id,gene,mutation = row['DepMap_ID'],row['Hugo_Symbol'],row['mutation']
			ccle2gens[ccle_id].add(gene)
			mutation = "@".join(mutation.split(" "))
			if mutation in self.annovar_dict:
				ccle2muts[ccle_id].add(mutation)
			else:
				ccle2muts[ccle_id].add('')
		# print(len(ccle2gens),len(ccle2muts))
		cols = ['mutation','sift_pred','sift4g_pred','lrt_pred','mutationtaster_pred','mutationassessor_pred','fathmm_pred','provean_pred','metasvm_pred','m_cap_pred',
					'primateai_pred','deogen2_pred','bayesdel_addaf_pred','bayesdel_noaf_pred','clinpred_pred','list_s2_pred','fathmm_mkl_coding_pred','fathmm_xf_coding_pred',
					'CLNSIG','GPD_unit']
		annovars = raw_ccle[cols]
		annovars["CLNSIG"]=annovars["CLNSIG"].fillna('None')
		annovars['category'] = annovars['CLNSIG'].apply(lambda x:clnsigs2cate[x])
		# print(annovars)

		annovars.insert(annovars.shape[1]-1, 'clinvar_Pathogenic', 0)
		annovars.insert(annovars.shape[1]-1, 'clinvar_Benign', 0)
		annovars.insert(annovars.shape[1]-1, 'clinvar_Unknown', 0)
		categories = ['Pathogenic','Benign','Unknown']
		for idx,row in annovars.iterrows():
			cate = row['category']
			if cate in categories:
				annovars.at[idx, 'clinvar_'+cate] = 1
		# print(annovars)
		annovars.insert(annovars.shape[1]-1, 'gpd_LU', 0)
		annovars.insert(annovars.shape[1]-1, 'gpd_PIU', 0)
		annovars.insert(annovars.shape[1]-1, 'gpd_NCU', 0)
		categories = ['PIU','LU', 'NCU']
		for idx,row in annovars.iterrows():
			cate = row['GPD_unit']
			if cate in categories:
				annovars.at[idx, 'gpd_'+cate] = 1
		annovars_ccle = annovars.drop(columns=['CLNSIG','category','GPD_unit'])
		# annovars_ccle.set_index('mutation', inplace=True)
		# print(annovars_ccle)

		### TCGA
		raw_tcga = pd.read_csv(fpath_tcga_mutation)
		tcga2gens,tcga2muts = defaultdict(set),defaultdict(set)
		for idx,row in raw_tcga.iterrows():
			tcga_id,gene,mutation = row['submitter_id'],row['gene'],row['point_mutation']
			tcga2gens[tcga_id].add(gene)
			mutation = "@".join(mutation.split(" "))
			if mutation in self.annovar_dict:
				tcga2muts[tcga_id].add(mutation)
			else:
				tcga2muts[tcga_id].add('')
		# print(len(tcga2gens),len(tcga2muts))
		cols = ['point_mutation','sift_pred','sift4g_pred','lrt_pred','mutationtaster_pred','mutationassessor_pred','fathmm_pred','provean_pred','metasvm_pred','m_cap_pred',
					'primateai_pred','deogen2_pred','bayesdel_addaf_pred','bayesdel_noaf_pred','clinpred_pred','list_s2_pred','fathmm_mkl_coding_pred','fathmm_xf_coding_pred',
					'CLNSIG','GPD_unit']
		annovars = raw_tcga[cols]
		annovars["CLNSIG"]=annovars["CLNSIG"].fillna('None')
		annovars['category'] = annovars['CLNSIG'].apply(lambda x:clnsigs2cate[x])
		# print(annovars)

		annovars.insert(annovars.shape[1]-1, 'clinvar_Pathogenic', 0)
		annovars.insert(annovars.shape[1]-1, 'clinvar_Benign', 0)
		annovars.insert(annovars.shape[1]-1, 'clinvar_Unknown', 0)
		categories = ['Pathogenic','Benign','Unknown']
		for idx,row in annovars.iterrows():
			cate = row['category']
			if cate in categories:
				annovars.at[idx, 'clinvar_'+cate] = 1
		# print(annovars)
		annovars.insert(annovars.shape[1]-1, 'gpd_LU', 0)
		annovars.insert(annovars.shape[1]-1, 'gpd_PIU', 0)
		annovars.insert(annovars.shape[1]-1, 'gpd_NCU', 0)
		categories = ['PIU','LU', 'NCU']
		for idx,row in annovars.iterrows():
			cate = row['GPD_unit']
			if cate in categories:
				annovars.at[idx, 'gpd_'+cate] = 1
		annovars_tcga = annovars.drop(columns=['CLNSIG','category','GPD_unit'])
		# annovars_tcga.set_index('point_mutation', inplace=True)
		# print(annovars_tcga)

		annoval_ccle = torch.tensor(np.concatenate((np.ones((1,len(annovars_ccle.columns[1:])),dtype=np.int64),annovars_ccle[annovars_ccle.columns[1:]].values,),axis=0,))
		annodict_ccle = {"@".join(k.split(" ")):(v+1) for v, k in annovars_ccle.mutation.to_dict().items()} #12175
		annoval_tcga = torch.tensor(np.concatenate((np.ones((1,len(annovars_tcga.columns[1:])),dtype=np.int64),annovars_tcga[annovars_tcga.columns[1:]].values,),axis=0,))
		annodict_tcga = {"@".join(k.split(" ")):(v+1) for v, k in annovars_tcga.point_mutation.to_dict().items()} #12175
		print(annoval_ccle.shape,annoval_tcga.shape) #torch.Size([62313, 23]) torch.Size([16832, 23])
		# sys.exit()

		# max_length = max([max([len(val) for val in ccle2genes.values()]), max([len(val) for val in tcga2genes.values()])])
		# max_gens,max_muts = max([len(val) for val in tcga2gens.values()]),max([len(val) for val in tcga2muts.values()])
		max_gens = max([max([len(val) for val in ccle2gens.values()]), max([len(val) for val in tcga2gens.values()])])
		max_muts = max([max([len(val) for val in ccle2muts.values()]), max([len(val) for val in tcga2muts.values()])])
		max_length = max_gens + max_muts
		# print(max_gens,max_muts,max_length)
		# ccle2genes_mat,ccle2genes_mask = np.zeros((len(ccle2genes),max_length),int),np.zeros((len(ccle2genes),max_length),int)
		ccle2gens_mat,ccle2muts_mat,ccle_mask = np.zeros((len(ccle2gens),max_gens),int),np.zeros((len(ccle2muts),max_muts),int),np.zeros((len(ccle2gens),max_length),int)
		tcga2gens_mat,tcga2muts_mat,tcga_mask = np.zeros((len(tcga2gens),max_gens),int),np.zeros((len(tcga2muts),max_muts),int),np.zeros((len(tcga2gens),max_length),int)
		# print(tcga2gens_mat.shape,tcga2muts_mat.shape,tcga_mask.shape) #(596, 198) (596, 378) (596, 576)
		for i,genes in enumerate(ccle2gens.values()):
			for j,gene in enumerate(genes):
				ccle2gens_mat[i,j] = self.gene2id[gene]
				ccle_mask[i,j] = 1
		for i,genes in enumerate(tcga2gens.values()):
			for j,gene in enumerate(genes):
				tcga2gens_mat[i,j] = self.gene2id[gene]
				tcga_mask[i,j] = 1 #tcga2genes_mask[i,j] = 1
		for i,muts in enumerate(ccle2muts.values()):
			for j,mut in enumerate(muts):
				if mut in annodict_ccle:
					ccle2muts_mat[i,j] = annodict_ccle[mut]
					ccle_mask[i,max_gens+j] = 1
		for i,muts in enumerate(tcga2muts.values()):
			for j,mut in enumerate(muts):
				if mut in annodict_tcga:
					tcga2muts_mat[i,j] = annodict_tcga[mut]
					tcga_mask[i,max_gens+j] = 1
		ccle2gens_mat_pd = pd.DataFrame(ccle2gens_mat,index=ccle2gens.keys())
		ccle2muts_mat_pd = pd.DataFrame(ccle2muts_mat,index=ccle2muts.keys()) #[596 rows x 378 columns]
		ccle_mask_pd = pd.DataFrame(ccle_mask,index=ccle2gens.keys())
		tcga2gens_mat_pd = pd.DataFrame(tcga2gens_mat,index=tcga2gens.keys()) #[596 rows x 198 columns]
		tcga2muts_mat_pd = pd.DataFrame(tcga2muts_mat,index=tcga2muts.keys()) #[596 rows x 378 columns]
		tcga_mask_pd = pd.DataFrame(tcga_mask,index=tcga2gens.keys()) #[596 rows x 576 columns]
		# return ccle2genes_mat_pd,ccle2genes_mask_pd,tcga2genes_mat_pd,tcga2genes_mask_pd
		# sys.exit()
		# return ccle2genes_mat_pd,ccle2genes_mask_pd,tcga2gens_mat_pd,tcga2muts_mat_pd,tcga_mask_pd
		return ccle2gens_mat_pd,ccle2muts_mat_pd,ccle_mask_pd,tcga2gens_mat_pd,tcga2muts_mat_pd,tcga_mask_pd,annoval_ccle,annoval_tcga


	def _load_mutation(self,):
		# fpath_ccle_mutation = '/data/hansheng/DruID/data/processed/ccle_raw_mutation.csv'
		fpath_ccle_mutation = '/data/hansheng/DruID/data/processed/ccle_21q3_annovar_gpd_annot_per_patient_per_mutation.csv'
		fpath_tcga_mutation = '/data/hansheng/DruID/data/processed/tcga_annovar_gpd_annot_per_patient_per_mutation.csv'

		# ccle_mutations = pd.read_csv(fpath_ccle_mutation) #[692 rows x 325 columns]
		# ccle_mutations.set_index('depmap_id', inplace=True) #[692 rows x 324 columns]
		# # print(ccle_mutations)
		# genes = ccle_mutations.columns.values.tolist() #324
		# depmaps = ccle_mutations.index.values.tolist() #692
		# values = ccle_mutations.to_numpy()
		# # print(values.shape)
		# ccle2genes = defaultdict(set)
		# rows,cols = np.where(values==1)
		# for idx in range(len(rows)):
		# 	ccle2genes[depmaps[rows[idx]]].add(genes[cols[idx]])

		raw_ccle = pd.read_csv(fpath_ccle_mutation)
		# print(raw_ccle)
		ccle2gens,ccle2muts = defaultdict(set),defaultdict(set)
		for idx,row in raw_ccle.iterrows():
			ccle_id,gene,mutation = row['DepMap_ID'],row['Hugo_Symbol'],row['mutation']
			ccle2gens[ccle_id].add(gene)
			mutation = "@".join(mutation.split(" "))
			if mutation in self.annovar_dict:
				ccle2muts[ccle_id].add(mutation)
			else:
				ccle2muts[ccle_id].add('')
		# print(len(ccle2gens),len(ccle2muts))

		raw_tcga = pd.read_csv(fpath_tcga_mutation)
		tcga2gens,tcga2muts = defaultdict(set),defaultdict(set)
		for idx,row in raw_tcga.iterrows():
			tcga_id,gene,mutation = row['submitter_id'],row['gene'],row['point_mutation']
			tcga2gens[tcga_id].add(gene)
			# tcga2muts[tcga_id].add(mutation)
			mutation = "@".join(mutation.split(" "))
			if mutation in self.annovar_dict:
				tcga2muts[tcga_id].add(mutation)
			else:
				tcga2muts[tcga_id].add('')
		# print(len(tcga2gens),len(tcga2muts))
		# print(raw_tcga)
		# sys.exit()

		# max_length = max([max([len(val) for val in ccle2genes.values()]), max([len(val) for val in tcga2genes.values()])])
		# max_gens,max_muts = max([len(val) for val in tcga2gens.values()]),max([len(val) for val in tcga2muts.values()])
		max_gens = max([max([len(val) for val in ccle2gens.values()]), max([len(val) for val in tcga2gens.values()])])
		max_muts = max([max([len(val) for val in ccle2muts.values()]), max([len(val) for val in tcga2muts.values()])])
		max_length = max_gens + max_muts
		# print(max_gens,max_muts,max_length)
		# ccle2genes_mat,ccle2genes_mask = np.zeros((len(ccle2genes),max_length),int),np.zeros((len(ccle2genes),max_length),int)
		ccle2gens_mat,ccle2muts_mat,ccle_mask = np.zeros((len(ccle2gens),max_gens),int),np.zeros((len(ccle2muts),max_muts),int),np.zeros((len(ccle2gens),max_length),int)
		tcga2gens_mat,tcga2muts_mat,tcga_mask = np.zeros((len(tcga2gens),max_gens),int),np.zeros((len(tcga2muts),max_muts),int),np.zeros((len(tcga2gens),max_length),int)
		# print(tcga2gens_mat.shape,tcga2muts_mat.shape,tcga_mask.shape) #(596, 198) (596, 378) (596, 576)
		for i,genes in enumerate(ccle2gens.values()):
			for j,gene in enumerate(genes):
				ccle2gens_mat[i,j] = self.gene2id[gene]
				ccle_mask[i,j] = 1
		for i,genes in enumerate(tcga2gens.values()):
			for j,gene in enumerate(genes):
				tcga2gens_mat[i,j] = self.gene2id[gene]
				tcga_mask[i,j] = 1 #tcga2genes_mask[i,j] = 1
		for i,muts in enumerate(ccle2muts.values()):
			for j,mut in enumerate(muts):
				if mut in self.annovar_dict:
					ccle2muts_mat[i,j] = self.annovar_dict[mut]
					ccle_mask[i,max_gens+j] = 1
		for i,muts in enumerate(tcga2muts.values()):
			for j,mut in enumerate(muts):
				if mut in self.annovar_dict:
					tcga2muts_mat[i,j] = self.annovar_dict[mut]
					tcga_mask[i,max_gens+j] = 1
		ccle2gens_mat_pd = pd.DataFrame(ccle2gens_mat,index=ccle2gens.keys())
		ccle2muts_mat_pd = pd.DataFrame(ccle2muts_mat,index=ccle2muts.keys()) #[596 rows x 378 columns]
		ccle_mask_pd = pd.DataFrame(ccle_mask,index=ccle2gens.keys())
		tcga2gens_mat_pd = pd.DataFrame(tcga2gens_mat,index=tcga2gens.keys()) #[596 rows x 198 columns]
		tcga2muts_mat_pd = pd.DataFrame(tcga2muts_mat,index=tcga2muts.keys()) #[596 rows x 378 columns]
		tcga_mask_pd = pd.DataFrame(tcga_mask,index=tcga2gens.keys()) #[596 rows x 576 columns]
		# return ccle2genes_mat_pd,ccle2genes_mask_pd,tcga2genes_mat_pd,tcga2genes_mask_pd
		# sys.exit()
		# return ccle2genes_mat_pd,ccle2genes_mask_pd,tcga2gens_mat_pd,tcga2muts_mat_pd,tcga_mask_pd
		return ccle2gens_mat_pd,ccle2muts_mat_pd,ccle_mask_pd,tcga2gens_mat_pd,tcga2muts_mat_pd,tcga_mask_pd


	def load_annotation(self, ):
		path = '/data/hansheng/DruID/data/pretrain/'
		gene324 = set(pd.read_table(path + "gene2ind.txt", header=None)[0])
		### GENIE CRC
		samples_crc = pd.read_csv(path + "samples_crc_genie.csv") #[5844 rows x 24 columns]
		samples_crc['tt_pfs_m_g_mos'] = samples_crc['tt_pfs_m_g_mos'].apply(lambda x: x*30)
		annovar_crc = pd.read_csv(path + "anno_features_per_mutation_crc_genie.csv")
		# print(annovar_crc)
		annovar_crc["count_flag"] = annovar_crc.drop(["input","gene"], axis=1).sum(axis=1)
		annovar_crc_max = (annovar_crc.drop_duplicates().groupby(by="input")[["count_flag"]].max().merge(annovar_crc,
				on=["input","count_flag"]).drop_duplicates(subset=["input","count_flag"])).reset_index(drop=True) #[12175 rows x 20 columns]
		# annovar_crc_max['input'] = annovar_crc_max['input'].apply(lambda x:"@".join(x.split(" ")))
		# print(samples_crc, annovar_crc_max)
		# print(annovar_crc_max)
		clinvar_crc = pd.read_csv(path + "clinvar_anno_features_per_mutation_genie.csv")
		# print(clinvar_crc)
		# clnsigs = list(set(clinvar_crc["CLNSIG"].to_list()))
		# print(clnsigs)
		clnsigs2cate = {'Pathogenic':'Pathogenic','Pathogenic|drug_response|other':'Pathogenic','Pathogenic/Likely_pathogenic':'Pathogenic',
						'Likely_pathogenic':'Pathogenic','Pathogenic/Likely_pathogenic|other':'Pathogenic','drug_response':'Pathogenic',
						'Likely_pathogenic|other':'Pathogenic','Pathogenic|risk_factor':'Pathogenic','Pathogenic/Likely_pathogenic|drug_response':'Pathogenic',
						'Likely_risk_allele':'Pathogenic','risk_factor':'Pathogenic','Likely_benign':'Benign','Benign/Likely_benign':'Benign','Benign':'Benign',
						'.':'Unknown','Uncertain_significance':'Unknown','Conflicting_interpretations_of_pathogenicity':'Unknown','not_provided':'Unknown',
						'Conflicting_interpretations_of_pathogenicity|other':'Unknown','Uncertain_significance|drug_response':'Unknown','other':'Unknown',
						'Affects':'---'}
		clinvar_crc['category'] = clinvar_crc['CLNSIG'].apply(lambda x:clnsigs2cate[x])
		# print(clinvar_crc)
		annovar_crc_max.insert(annovar_crc_max.shape[1]-1, 'clinvar_Pathogenic', 0)
		annovar_crc_max.insert(annovar_crc_max.shape[1]-1, 'clinvar_Benign', 0)
		annovar_crc_max.insert(annovar_crc_max.shape[1]-1, 'clinvar_Unknown', 0)
		categories = ['Pathogenic','Benign','Unknown']
		clinvar2cate = {row['input']:row['category'] for _,row in clinvar_crc.iterrows()}
		for idx,row in annovar_crc_max.iterrows():
			if row['input'] in clinvar2cate.keys():
				cate = clinvar2cate[row['input']]
				if cate in categories:
					annovar_crc_max.at[idx, 'clinvar_'+cate] = 1
		# print(annovar_crc_max)

		gpd_crc = pd.read_csv(path + "gpd_anno_features_per_mutation_genie.csv")
		annovar_crc_max.insert(annovar_crc_max.shape[1]-1, 'gpd_LU', 0)
		annovar_crc_max.insert(annovar_crc_max.shape[1]-1, 'gpd_PIU', 0)
		annovar_crc_max.insert(annovar_crc_max.shape[1]-1, 'gpd_NCU', 0)
		categories = ['PIU','LU', 'NCU']
		gpd2cate = {row['point_mutations_canonicalized']:row['GPD_unit'] for _,row in gpd_crc.iterrows()}
		for idx,row in annovar_crc_max.iterrows():
			if row['input'] in gpd2cate.keys():
				cate = gpd2cate[row['input']]
				annovar_crc_max.at[idx, 'gpd_'+cate] = 1
		# print(annovar_crc_max)
		# sys.exit()

		### NSCLC
		samples_clc = pd.read_csv(path + "samples_nsclc.csv").drop(["Unnamed: 0"], axis=1) #[14910 rows x 24 columns]
		# samples_clc['tt_pfs_m_g_mos'] = samples_clc['tt_pfs_m_g_mos'].apply(lambda x: x/30)
		annovar_clc = pd.read_csv(path + "annovar_features_per_mutation_nsclc.csv")
		annovar_clc["count_flag"] = annovar_crc.drop(["input","gene"], axis=1).sum(axis=1)
		annovar_clc_max = (annovar_clc.drop_duplicates().groupby(by="input")[["count_flag"]].max().merge(annovar_clc,
				on=["input","count_flag"]).drop_duplicates(subset=["input","count_flag"])).reset_index(drop=True) #[1058 rows x 20 columns]
		# annovar_clc_max['input'] = annovar_clc_max['input'].apply(lambda x:"@".join(x.split(" ")))
		# print(samples_clc, annovar_clc_max)
		clinvar_clc = pd.read_csv(path + "clinvar_anno_features_per_mutation_nsclc.csv")
		# print(clinvar_clc)
		clnsigs = list(set(clinvar_clc["CLNSIG"].to_list()))
		clinvar_clc['category'] = clinvar_clc['CLNSIG'].apply(lambda x:clnsigs2cate[x])
		# print(clinvar_clc)

		annovar_clc_max.insert(annovar_clc_max.shape[1]-1, 'clinvar_Pathogenic', 0)
		annovar_clc_max.insert(annovar_clc_max.shape[1]-1, 'clinvar_Benign', 0)
		annovar_clc_max.insert(annovar_clc_max.shape[1]-1, 'clinvar_Unknown', 0)
		categories = ['Pathogenic','Benign','Unknown']
		clinvar2cate = {row['input']:row['category'] for _,row in clinvar_clc.iterrows()}
		for idx,row in annovar_clc_max.iterrows():
			if row['input'] in clinvar2cate.keys():
				cate = clinvar2cate[row['input']]
				if cate in categories:
					annovar_clc_max.at[idx, 'clinvar_'+cate] = 1
		# print(annovar_clc_max)
		gpd_clc = pd.read_csv(path + "nsclc_anno_features_per_mutation_astar.csv")
		annovar_clc_max.insert(annovar_clc_max.shape[1]-1, 'gpd_LU', 0)
		annovar_clc_max.insert(annovar_clc_max.shape[1]-1, 'gpd_PIU', 0)
		annovar_clc_max.insert(annovar_clc_max.shape[1]-1, 'gpd_NCU', 0)
		categories = ['PIU','LU','NCU']
		gpd2cate = {row['point_mutations_canonicalized']:row['GPD_unit'] for _,row in gpd_clc.iterrows()}
		for idx,row in annovar_clc_max.iterrows():
			if row['input'] in gpd2cate.keys():
				cate = gpd2cate[row['input']]
				annovar_clc_max.at[idx, 'gpd_'+cate] = 1
		# print(annovar_clc_max)

		# sys.exit()
		### ADD DRUGS INFO
		survival_clc = pd.read_csv(path + "patient_survival_info_nsclc.csv")
		patient2drug = dict()
		for idx,row in survival_clc.iterrows():
			patient2drug[row['patient_id']] = row['drug_name']
		samples_clc['drug_name'] = samples_clc['sample_id'].apply(lambda x:patient2drug[x])
		samples_crc['drug_name'] = samples_crc['sample_id'].apply(lambda x:'5-Fluorouracil')
		### MARGE CRC and CLC
		samples = pd.concat([samples_crc,samples_clc])
		annovar_max = pd.concat([annovar_crc_max,annovar_clc_max])
		annovar_max = (annovar_max.drop_duplicates().groupby(by="input")[["count_flag"]].max().merge(annovar_max,
				on=["input","count_flag"]).drop_duplicates(subset=["input","count_flag"])).reset_index(drop=True) #[1058 rows x 20 columns]
		### MAP DRUGS TO ID
		# drugs = list(set(samples['drug_name'].to_numpy())) #14+1
		# self.drug2id = dict(zip(drugs, [i for i in range(len(drugs))]))
		# print(self.drug2id)
		# samples['drug_name'] = samples['drug_name'].apply(lambda x: self.drug2id[x])
		# sys.exit()
		self.samples_clc = samples_clc
		self.annovar_clc_max = annovar_clc_max
		# print(annovar_clc_max)
		# sys.exit()
		return samples, gene324, annovar_max
		# return samples_clc,gene324,annovar_crc_max

	def get_texts(self, samples):
		texts = (samples.groupby(["sample_id","patient_id","drug_name","tt_pfs_m_g_mos","pfs_m_g_status","Hugo_Symbol",])["mutation"].agg(lambda x: " ".join(["@".join(y.split(" ")) for y in list(x)]))
			.reset_index(level="Hugo_Symbol").apply(lambda x: x["Hugo_Symbol"] + " <mutsep> " + x["mutation"], axis=1).groupby(["sample_id","patient_id","drug_name","tt_pfs_m_g_mos","pfs_m_g_status"]).agg(lambda x: " <gensep> ".join(list(x))))
		return texts

	def yield_tokens(self, texts, genes):
		geneset = genes.union({"<gensep>", "<mutsep>"}) #326
		for text in texts:
			tokens = ["<unk>" if not "@" in tok and not tok in geneset else tok for tok in text.split(" ")]
			univariate_mutation_tokens = ["<mut>" if "@" in tok else tok for tok in tokens]  # Treat all mutations to have the same token <mut>
			yield univariate_mutation_tokens
		# KRAS <mutsep> KRAS@G12S <gensep> TP53 <mutsep> TP53@R175H
		# ['KRAS', '<mutsep>', 'KRAS@G12S', '<gensep>', 'TP53', '<mutsep>', 'TP53@R175H']
		# ['KRAS', '<mutsep>', '<mut>', '<gensep>', 'TP53', '<mutsep>', '<mut>']

	# def process_celline(self, data, mutations, masks, drug_fp):
	def process_celline(self, data, mutations, annovar, masks, drug_fp):
		cl_features,cl_y = [],[]
		for idx, row in data.y_df.iterrows(): #depmap_id  ACH-000805 \n drug_name  CISPLATIN \n auc  0.758212
			row_inp = []
			# row_inp.extend(data.clinvar_gpd_annovar_annotated.loc[row["depmap_id"]].values)
			row_inp.extend(mutations.loc[row["depmap_id"]].values)
			row_inp.extend(annovar.loc[row["depmap_id"]].values)
			row_inp.extend(masks.loc[row["depmap_id"]].values)
			row_inp.extend(drug_fp.loc[row["drug_name"]].values)
			row_inp.append(row["auc"])
			cl_y.append(row["auc"])
			cl_features.append(row_inp)
			# sys.exit()
		return cl_features,cl_y

	def process_tcga(self, data, mutations, annovar, masks, drug_fp):
		tcga_features,tcga_y = [],[]
		# tcga_events,tcga_times = [],[]
		for idx, row in data.tcga_response.iterrows(): #submitter_id TCGA-EO-A1Y5 \n drug_name  PACLITAXEL \n response  1
			row_inp = []
			row_inp.extend(mutations.loc[row["submitter_id"]].values) #243
			# print(mutations.loc[row["submitter_id"]].values.shape)
			row_inp.extend(annovar.loc[row["submitter_id"]].values) #38
			# print(annovar.loc[row["submitter_id"]].values.shape)
			row_inp.extend(masks.loc[row["submitter_id"]].values) #281
			# print(masks.loc[row["submitter_id"]].values.shape)
			row_inp.extend(drug_fp.loc[row["drug_name"]].values) #2048
			# row_inp.extend(self.drug2id[row["drug_name"]]) #new
			row_inp.append(row["response"])
			tcga_y.append(row["response"])
			tcga_features.append(row_inp)
			# sys.exit()
		return tcga_features,tcga_y#,tcga_times,tcga_events


class CustomDataset_cl(Dataset):
	def __init__(self, train_features):
		self.train_features = train_features

	def __len__(self):
		return len(self.train_features)

	def __getitem__(self, idx):
		# return torch.Tensor(self.train_features[idx][:-1]), self.train_features[idx][-1]
		# return torch.Tensor(self.train_features[idx][:576]),torch.Tensor(self.train_features[idx][576:1152]),torch.Tensor(self.train_features[idx][1152:-1]), self.train_features[idx][-1]
		return torch.Tensor(self.train_features[idx][:243]),torch.Tensor(self.train_features[idx][243:281]),\
				torch.Tensor(self.train_features[idx][281:562]),torch.Tensor(self.train_features[idx][562:-1]),self.train_features[idx][-1]

class CustomDataset_tcga(Dataset):
	def __init__(self, train_features):
		self.train_features = train_features

	def __len__(self):
		return len(self.train_features)

	def __getitem__(self, idx):
		# return torch.Tensor(self.train_features[idx][:-1]), self.train_features[idx][-1]
		# return torch.Tensor(self.train_features[idx][:222]),torch.Tensor(self.train_features[idx][222:444]),torch.Tensor(self.train_features[idx][444:-1]), self.train_features[idx][-1]
		return torch.Tensor(self.train_features[idx][:243]),torch.Tensor(self.train_features[idx][243:281]),\
				torch.Tensor(self.train_features[idx][281:562]),torch.Tensor(self.train_features[idx][562:-1]),self.train_features[idx][-1]

class CustomDataset_survival(Dataset):
	def __init__(self, train_features,train_targets):
		self.train_features = train_features
		self.train_targets = train_targets

	def __len__(self):
		return len(self.train_features)

	def __getitem__(self, idx):
		return torch.Tensor(self.train_features[idx][:44]),torch.Tensor(self.train_features[idx][44:88]),torch.Tensor(self.train_features[idx][88:]),self.train_targets[idx]

# --- Cell Line Dataset --- #
class CellLineDataset(Dataset):
	# Base class for datasets that hold cell line information
	base_dir = "../"
	entity_identifier_name = None

	def __str__(self):
		dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
		return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
		#Entities - {len(dataset_df[self.entity_identifier_name].unique())}
		#Drugs - {len(dataset_df.drug_name.unique())}
		#Pairs - {len(self)}
		"""
	pass

class CategoricalAnnotatedCellLineDatasetFilteredByDrug(CellLineDataset):
	# Cell line data with categorical annotation features from annovar
	entity_identifier_name = "depmap_id"

	def __init__(self, is_train=True, filter_for="rad51", xon17=False, sample_id=0):
		"""
		is_train : bool
			Returns items from the train or test split (defaults to True)
		filter_for : str
			Filters for drugs in the dataset passed (defaults to "rad51"). Can also take in value "tcga" or "nuh_crc"
		xon17 : bool
			Uses the variant annotation that returns the score for each gene as 1 + x/17 (defaults to False)
		"""
		self.is_train = is_train
		self.sample_id = sample_id
		if xon17 == False:
			self.df_reprn_mut = pd.read_csv("../data/processed/ccle_anno_features.csv",)
		else:
			raise("xon17 not implemented for ccle_anno_features.csv")
		self.df_reprn_mut.set_index(["depmap_id"], inplace=True)

		# Check if there are any cell-lines for which there are no valid mutations and filter those out
		mask = True
		for col in self.df_reprn_mut.columns:
			mask = mask & (self.df_reprn_mut[col] == 0)

		self.df_reprn_mut = self.df_reprn_mut[~mask].copy()

		df_auc = pd.read_csv("../data/raw/cell_drug_auc_final_1111.csv")
		df_auc["depmap_id"] = df_auc["ARXSPAN_ID"].astype("string")
		df_auc.drop("ARXSPAN_ID", axis=1, inplace=True)
		df_auc.set_index(["depmap_id"], inplace=True)

		# Filter for Rad51 drugs only
		self.filter_for = filter_for
		if self.filter_for == "rad51":
			list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOXORUBICIN"]
		elif self.filter_for == "tcga":
			list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOCETAXEL", "5-FLUOROURACIL", "CYCLOPHOSPHAMIDE"]
		elif self.filter_for == "nuh_crc":
			list_drugs = ["5-FLUOROURACIL", "OXALIPLATIN", "IRINOTECAN", "CETUXIMAB"]
		filtered_drugs = [col for col in df_auc.columns if col in list_drugs]
		df_auc = df_auc[filtered_drugs]

		train_cell_lines_ids = pd.read_csv(f"../data/raw/train_celllines_filtered4{filter_for}_drugs_sample{sample_id}.csv",header=None)[0].values
		test_cell_lines_ids = pd.read_csv(f"../data/raw/test_celllines_filtered4{filter_for}_drugs_sample{sample_id}.csv",header=None)[0].values

		if is_train is not None:
			if self.is_train:
				required_cell_line_ids = train_cell_lines_ids
			else:
				required_cell_line_ids = test_cell_lines_ids
		else:
			required_cell_line_ids = np.concatenate([train_cell_lines_ids, test_cell_lines_ids])
		
		y_df = df_auc[df_auc.index.isin(required_cell_line_ids)].copy()

		# The below filter is to remove those cellines for which there are no annotation features available (likely due to the absence of point mutations in such cases)
		# TODO: Check how to represent those cases that do not have any point mutations
		y_df = y_df[y_df.index.isin(self.df_reprn_mut.index.get_level_values(0))].copy()
		y_df = y_df.reset_index().melt(id_vars=["depmap_id",], var_name="drug_name", value_name="auc",)
		# When scaling is not done, filter those entries with value -99999
		self.y_df = y_df[~(y_df.auc < 0)]

	def __len__(self):
		return len(self.y_df)

	def __getitem__(self, idx):
		record = self.y_df.iloc[idx]
		return {"depmap_id": record["depmap_id"],"drug_name": record["drug_name"],"auc": record["auc"],}

	@cached_property
	def mutations(self):
		return self.df_reprn_mut

	@cached_property
	def raw_mutations(self):
		df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
		df_reprn_mut.set_index(["depmap_id"], inplace=True)

		# Check if there are any cell-lines for which there are no valid mutations and filter those out
		mask = True
		for col in df_reprn_mut.columns:
			mask = mask & (df_reprn_mut[col] == 0)

		df_reprn_mut = df_reprn_mut[~mask].copy()
		df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
		return df_reprn_mut

	@cached_property
	def raw_mutations_all_genes(self):
		df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation_all_genes.csv",)
		df_reprn_mut.set_index(["depmap_id"], inplace=True)

		# Check if there are any cell-lines for which there are no valid mutations and filter those out
		mask = True
		for col in df_reprn_mut.columns:
			mask = mask & (df_reprn_mut[col] == 0)

		df_reprn_mut = df_reprn_mut[~mask].copy()
		df_reprn_mut = df_reprn_mut.reindex(columns=ALL_CCLE_GENES)
		return df_reprn_mut
	
	@cached_property
	def raw_mutations_285_genes(self):
		df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
		df_reprn_mut.set_index(["depmap_id"], inplace=True)

		# Check if there are any cell-lines for which there are no valid mutations and filter those out
		mask = True
		for col in df_reprn_mut.columns:
			mask = mask & (df_reprn_mut[col] == 0)

		df_reprn_mut = df_reprn_mut[~mask].copy()
		df_reprn_mut = df_reprn_mut[GENES_285]
		df_reprn_mut = df_reprn_mut.reindex(columns=GENES_285)
		return df_reprn_mut

	@cached_property
	def embedded_raw_mutations_all_genes(self):
		# 324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder
		df_reprn_mut_train = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_train.csv", index_col = 0)
		df_reprn_mut_test = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_test.csv", index_col = 0)
		return pd.concat([df_reprn_mut_test, df_reprn_mut_train])

	@cached_property
	def embedded_raw_mutations_all_genes_v2(self):
		# 324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder with more training
		df_reprn_mut_train = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_train_v2.csv", index_col = 0)
		df_reprn_mut_test = pd.read_csv(f"../data/processed/ccle_embedded_all_genes_324_sample{self.sample_id}_test_v2.csv", index_col = 0)
		return pd.concat([df_reprn_mut_test, df_reprn_mut_train])   

	@cached_property
	def gene_exp(self):
		# 324 dimensional gene expression values for F1 genes
		gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0)
		return gene_exp_df
	
	@cached_property
	def gene_exp_285(self):
		# 285 dimensional intersecting gene set across Tempus, TruSight, F1
		gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0)
		return gene_exp_df[GENES_285]

	@cached_property
	def gene_exp_1426(self):
		# CODE-AE genes' gene expression
		gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression_code_ae_genes.csv", index_col=0)
		indices = list(pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0).index)
		return gene_exp_df.loc[indices]

	@cached_property
	def gene_exp_1426_codeae_sample(self):
		# gene expression for 1426 genes in CODE-AE alongwith all samples used in CODE-AE
		gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression_code_ae_genes_and_samples_scaled.csv", index_col = 0)
		gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["depmap_id"]).set_index("depmap_id", drop=True)
		return gene_exp_df

	@cached_property
	def cnv(self):
		cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)
		return cnv_df

	@cached_property
	def cnv_285(self):
		cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)
		return cnv_df[modified_GENES_285]

	@cached_property
	def concatenated_raw_mutation_cnv(self):
		# raw mutations 
		df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
		df_reprn_mut.set_index(["depmap_id"], inplace=True)

		# Check if there are any cell-lines for which there are no valid mutations and filter those out
		mask = True
		for col in df_reprn_mut.columns:
			mask = mask & (df_reprn_mut[col] == 0)

		df_reprn_mut = df_reprn_mut[~mask].copy()
		df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
		cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)

		return pd.concat([df_reprn_mut, cnv_df], axis = 1)

	@cached_property
	def concatenated_raw_mutation_cnv_285(self):
		# raw mutations 
		df_reprn_mut = pd.read_csv("../data/processed/ccle_raw_mutation.csv",)
		df_reprn_mut.set_index(["depmap_id"], inplace=True)

		# Check if there are any cell-lines for which there are no valid mutations and filter those out
		mask = True
		for col in df_reprn_mut.columns:
			mask = mask & (df_reprn_mut[col] == 0)

		df_reprn_mut = df_reprn_mut[~mask].copy()
		df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
		cnv_df = pd.read_csv("../data/processed/ccle_cnv_actual_ohe_encoded.csv", index_col=0)
		return pd.concat([df_reprn_mut, cnv_df], axis = 1)[GENES_285 + modified_GENES_285]

	@cached_property
	def clinvar_gpd_annovar_annotated(self):
		clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_ccle_feature_matrix.csv", index_col = 0)
		return clinvar_gpd_annovar_df
	
	@cached_property
	def clinvar_gpd_annovar_annotated_285(self):
		clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_ccle_feature_matrix.csv", index_col = 0)
		return clinvar_gpd_annovar_df[modified_GENES_285_clinvar]
	
	@cached_property
	def concatenated_anno_mutation_gene_exp(self):
		gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv", index_col = 0)
		clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_ccle_feature_matrix.csv")
		clinvar_gpd_annovar_df.rename(columns={"DepMap_ID": "depmap_id"}, inplace=True)
		clinvar_gpd_annovar_df.set_index("depmap_id", drop=True, inplace=True)
		return pd.concat([gene_exp_df, clinvar_gpd_annovar_df], axis = 1)


class AggCategoricalAnnotatedCellLineDatasetFilteredByDrug(CategoricalAnnotatedCellLineDatasetFilteredByDrug):
	# Cell line data with categorical annotation features from annovar aggregated per gene
	@cached_property
	def mutations(self):
		agg_results = {}
		for gene in GENES_324:
			filtered_df = self.df_reprn_mut.filter(regex=f"^{gene}_[a-z]*")
			# agg_results[gene] = filtered_df.mean(axis=1)
			# agg_results[gene] = filtered_df.sum(axis=1)
			curr_result = None
			for col in filtered_df.columns:
				if type(curr_result) == pd.Series:
					curr_result = curr_result | (filtered_df[col] != 0)
				else:
					curr_result = filtered_df[col] != 0

			agg_results[gene] = curr_result.astype(np.int32)

		agg_df = pd.DataFrame(agg_results)
		return agg_df

	@cached_property
	def drug_repr(self):
		drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
		drug_fp_df.set_index("drug_name", inplace=True)
		return drug_fp_df


# --- TCGA Dataset --- #
class TcgaDataset(Dataset):
	# Base class for datasets that hold TCGA information
	def __str__(self):
		dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
		return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
		#Entities - {len(dataset_df[self.entity_identifier_name].unique())}
		#Drugs - {len(dataset_df.drug_name.unique())}
		#Pairs - {len(self)}
		#Response (0 to 1) - {dataset_df.response.value_counts()[0]} to {dataset_df.response.value_counts()[1]}
		"""
	pass

class CategoricalAnnotatedTcgaDatasetFilteredByDrug(TcgaDataset):
	# TCGA data, used only for testing
	entity_identifier_name = "submitter_id"

	def __init__(self, is_train=False,filter_for="tcga",xon17=False,sample_id=0):
		self.is_train = is_train
		self.sample_id = sample_id
		tcga_response = pd.read_csv("../data/processed/TCGA_drug_response_010222.csv")
		tcga_response.rename(columns={"patient.arr":self.entity_identifier_name,"drug":"drug_name","response":"response_description","response_cat":"response",},inplace=True,)

		if xon17 == False:
			tcga_mutation = pd.read_csv("../data/processed/tcga_anno_features_only_categorical_agg.csv")
		else:
			raise("xon17 not implemented for tcga_anno_features_only_categorical_agg.csv")
		tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
		self.tcga_mutation_filtered = tcga_mutation
		
		if filter_for=="tcga":
			list_drugs = ["CISPLATIN", "PACLITAXEL", "GEMCITABINE", "DOCETAXEL", "5-FLUOROURACIL", "CYCLOPHOSPHAMIDE"]
			tcga_response = tcga_response[tcga_response["drug_name"].isin(list_drugs)]
			tcga_response = tcga_response[tcga_response[self.entity_identifier_name].isin(tcga_mutation.index.get_level_values(0))]
			self.tcga_response = tcga_response[[self.entity_identifier_name, "drug_name", "response"]].copy()
			self.tcga_response = self.tcga_response[self.tcga_response.drug_name.isin(list_drugs)].reset_index(drop=True)
		elif filter_for=="nuh_crc":
			list_drugs = ["5-FLUOROURACIL", "OXALIPLATIN", "IRINOTECAN", "CETUXIMAB"]
			tcga_response = tcga_response[tcga_response["drug_name"].isin(list_drugs)]
			tcga_response = tcga_response[
			tcga_response[self.entity_identifier_name].isin(tcga_mutation.index.get_level_values(0))]
			self.tcga_response = tcga_response[[self.entity_identifier_name, "drug_name", "response"]].copy()
			self.tcga_response = self.tcga_response[self.tcga_response.drug_name.isin(list_drugs)].reset_index(drop=True)

		train_tcga_ids = pd.read_csv(f"../data/raw/train_tcga_filtered4tcga_drugs_sample{sample_id}.csv", header=None)[0].values
		test_tcga_ids = pd.read_csv(f"../data/raw/test_tcga_filtered4tcga_drugs_sample{sample_id}.csv", header=None)[0].values

		if is_train is not None:
			if self.is_train:
				required_tcga_ids = train_tcga_ids
			else:
				required_tcga_ids = test_tcga_ids
		else:
			required_tcga_ids = np.concatenate([train_tcga_ids, test_tcga_ids])
		
		self.tcga_response = self.tcga_response[self.tcga_response.submitter_id.isin(required_tcga_ids)].copy()
		
	def __len__(self):
		return len(self.tcga_response)

	def __getitem__(self, idx):
		record = self.tcga_response.iloc[idx]
		return {self.entity_identifier_name:record[self.entity_identifier_name],"drug_name":record["drug_name"],"response":record["response"],}

	@cached_property
	def mutations(self):
		return self.tcga_mutation_filtered

	@cached_property
	def raw_mutations(self):
		tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
		tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
		tcga_mutation = tcga_mutation.reindex(columns=GENES_324)
		return tcga_mutation
	
	@cached_property
	def raw_mutations_all_genes(self):
		tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222_all_genes")
		tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
		tcga_mutation = tcga_mutation.reindex(columns=ALL_CCLE_GENES)
		return tcga_mutation
	
	@cached_property
	def raw_mutations_285_genes(self):
		tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
		tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
		tcga_mutation = tcga_mutation[GENES_285].reindex(columns=GENES_285)
		return tcga_mutation
	
	@cached_property
	def embedded_raw_mutations_all_genes(self):
		# 324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder
		df_reprn_mut_train = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_train.csv", index_col = 0)
		df_reprn_mut_test = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_test.csv", index_col = 0)
		return pd.concat([df_reprn_mut_test, df_reprn_mut_train])	 
	
	@cached_property
	def embedded_raw_mutations_all_genes_v2(self):
		# 324 dimensional embedding of all 19536 genes, embedded using a regular Autoencoder with more training
		df_reprn_mut_train = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_train_v2.csv", index_col = 0)
		df_reprn_mut_test = pd.read_csv(f"../data/processed/tcga_embedded_all_genes_324_sample{self.sample_id}_test_v2.csv", index_col = 0)
		return pd.concat([df_reprn_mut_test, df_reprn_mut_train])

	@cached_property
	def gene_exp(self):
		# 324 dimensional gene expression values for F1 genes
		gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv", index_col = 0)
		gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
		return gene_exp_df
	
	@cached_property
	def gene_exp_285(self):
		# 285 dimensional intersecting gene set across Tempus, TruSight, F1
		gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv", index_col = 0)
		gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
		return gene_exp_df[GENES_285]
	
	@cached_property
	def gene_exp_1426(self):
		# gene expression values for the 1426 genes used in CODE-AE for our samples
		gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression_code_ae_genes.csv", index_col = 0)
		gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
		return gene_exp_df
	
	@cached_property
	def gene_exp_1426_codeae_sample(self):
		# gene expression for 1426 genes in CODE-AE alongwith all samples used in CODE-AE
		gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression_code_ae_genes_and_samples_scaled.csv", index_col = 0)
		gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).set_index("tcga_id", drop=True)
		return gene_exp_df
	
	@cached_property
	def cnv(self):
		tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
		return tcga_cnv
	
	@cached_property
	def cnv_285(self):
		tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
		return tcga_cnv[modified_GENES_285]

	@cached_property
	def concatenated_raw_mutation_cnv(self):
		# raw mutations
		tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
		tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
		tcga_mutation = tcga_mutation.reindex(columns=GENES_324)
		# cnv
		tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
		return pd.concat([tcga_mutation, tcga_cnv], axis = 1)
	
	@cached_property
	def concatenated_raw_mutation_cnv_285(self):
		# raw mutations
		tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
		tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
		tcga_mutation = tcga_mutation.reindex(columns=GENES_324)
		# cnv
		tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_actual_ohe_encoded", index_col=0)
		return pd.concat([tcga_mutation, tcga_cnv], axis = 1)[GENES_285 + modified_GENES_285]
	
	@cached_property
	def raw_mutation_tcga_msk_impact(self):
		# raw mutations TCGA
		tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
		tcga_mutation.rename(columns={self.entity_identifier_name: "patient_id"}, inplace=True)
		tcga_mutation.set_index("patient_id", drop=True, inplace=True)
		tcga_mutation = tcga_mutation.reindex(columns=GENES_324)
		# raw mutations MSK Impact
		msk_impact_mutation = pd.read_csv("../data/processed/msk_impact_mutations.csv", index_col=0)
		return pd.concat([tcga_mutation, msk_impact_mutation], axis = 0)

	@cached_property
	def survival_info(self):
		survival_info_df = pd.read_csv("../data/processed/survival_rate_final_010222")
		survival_info_df.rename(columns={"demographic.days_to_death": "days"}, inplace=True)
		survival_info_df.drop(columns=["demographic.vital_status", "days_to_death_scaled"], inplace=True)
		return survival_info_df
	
	@cached_property
	def clinvar_gpd_annovar_annotated(self):
		clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_tcga_feature_matrix.csv", index_col = 0)
		return clinvar_gpd_annovar_df

	@cached_property
	def clinvar_gpd_annovar_annotated_285(self):
		clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_tcga_feature_matrix.csv", index_col = 0)
		return clinvar_gpd_annovar_df[modified_GENES_285_clinvar]
	
	@cached_property
	def concatenated_anno_mutation_gene_exp(self):
		gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv", index_col = 0)
		gene_exp_df = gene_exp_df.reset_index().drop_duplicates(subset=["tcga_id"]).rename(columns={"tcga_id": "submitter_id"})
		gene_exp_df.set_index("submitter_id", drop=True, inplace=True)
		
		clinvar_gpd_annovar_df = pd.read_csv("../data/processed/clinvar_gpd_annovar_annotated_tcga_feature_matrix.csv", index_col = 0)
		return pd.concat([gene_exp_df, clinvar_gpd_annovar_df], axis = 1).fillna(0)


class AggCategoricalAnnotatedTcgaDatasetFilteredByDrug(CategoricalAnnotatedTcgaDatasetFilteredByDrug):
	# Aggregated categorical annotations features for TCGA entities
	@cached_property
	def mutations(self):
		agg_results = {}
		for gene in GENES_324:
			filtered_df = self.tcga_mutation_filtered.filter(regex=f"^{gene}_[a-z]*")
			# agg_results[gene] = filtered_df.mean(axis=1)
			# agg_results[gene] = filtered_df.sum(axis=1)
			curr_result = None
			for col in filtered_df.columns:
				if type(curr_result) == pd.Series:
					curr_result = curr_result | (filtered_df[col] != 0)
				else:
					curr_result = filtered_df[col] != 0

			agg_results[gene] = curr_result.astype(np.int32)

		agg_df = pd.DataFrame(agg_results)
		return agg_df

	@cached_property
	def drug_repr(self):
		drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
		drug_fp_df.set_index("drug_name", inplace=True)
		return drug_fp_df


class Tokenizer:
	# vocab: vocabulary instance; annovar: annovar dictionary {gene_indicator: gene_idx}
	def __init__(self, vocab, annovar) -> None:
		self.vocab = vocab
		self.annovar = annovar #{'ZRSR2@V250M': 12175} #12942

	def __call__(self,texts,padding=False,max_length:Optional[int]=None,return_tensors:Optional[Union[str,TensorType]]=None,return_attention_mask=True,):
		return self.batch_encode_plus(texts=texts,padding=padding,max_length=max_length,return_tensors=return_tensors,return_attention_mask=return_attention_mask,)

	def batch_encode_plus(self,texts,padding=False,max_length:Optional[int]=None,return_tensors:Optional[Union[str,TensorType]]=None,return_attention_mask=True,):
		if type(texts) == str:
			texts = [texts]
		masked_texts = [" ".join(["<mut>" if "@" in tok else tok for tok in text.split(" ")]) for text in texts] #['TP53 <mutsep> <mut>']
		batch_tokens = [["<s>"] + text.split(" ") + ["</s>"] for text in masked_texts] #['<s>','TP53','<mutsep>','<mut>','</s>']
		batch_tokens = [[token if token in self.vocab else "<unk>" for token in tokens] for tokens in batch_tokens] #['<s>','<unk>','<mutsep>','<mut>','<gensep>','TP53','<mutsep>','<mut>','</s>']
		batch_numerical_tokens = [self.vocab.lookup_indices(tokens) for tokens in batch_tokens] #[0,8,5,4,2]
		batch_attention = [[1] * len(tokens) for tokens in batch_tokens] 	#! 1=non-padding
		batch_annovar = [[0 if not tok in self.annovar else self.annovar[tok] for tok in tokens] for tokens in [["<s>"]+text.split(" ")+["</s>"] for text in texts]]
		# print(batch_annovar) #[0, 0, 0, 1579, 0, 0, 0, 11577, 0] #331

		if padding:
			if max_length is None:
				max_length = max([len(tokens) for tokens in batch_tokens]) #567
			for numerical_tokens,attention,annovar in zip(batch_numerical_tokens,batch_attention,batch_annovar):
				padding_num = max_length-len(numerical_tokens)
				if padding_num > 0:
					numerical_tokens += [1]*padding_num
					attention += [0]*padding_num
					annovar += [0]*padding_num

		if return_tensors == "pt":
			try:
				batch_numerical_tokens = torch.tensor(batch_numerical_tokens)
				batch_attention = torch.tensor(batch_attention)
				batch_annovar = torch.tensor(batch_annovar)
			except:
				raise ValueError("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length.")
			assert batch_annovar.max() <= len(self.annovar), "Check annovar dict consistency in indices"

		output = {"input_ids": batch_numerical_tokens,"attention_mask": batch_attention,"annovar_mask": batch_annovar,}
		if not return_attention_mask:
			output.pop("attention_mask")
		return BatchEncoding(output)
		



if __name__ == '__main__':
	data = MedicalDATA()
	print()
