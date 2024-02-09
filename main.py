# !/usr/bin/env python
# -*- coding: utf8 -*-

import argparse,sys
import loader as loader
import models as models
import utils as utils
import warnings
warnings.simplefilter('ignore')


def parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--hid_dim', type=int, default=64, help='Dimension of hidden.')
	parser.add_argument('--opt', type=str, default='Adam', help='String of optimizer.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Number of learning rate.')
	parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
	parser.add_argument('--dropout', type=float, default=0.1, help='Number of dropout.')
	parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
	parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping (# of epochs).')
	parser.add_argument('--sample', type=int, default=0, help='Number of sample.')
	parser.add_argument('--batchsize', type=int, default=128, help='Number of batch size.')
	parser.add_argument('--gpu', type=int, default=0, help='Number of GPU.')
	parser.add_argument('--seed', type=int, default=0, help='Number of SEED.')
	parser.add_argument('--model', type=str, default='TransformerMTLR', help='Model to train.')

	args = parser.parse_args()
	print(args)
	return args

def run(args):
	utils.seed_torch(args.seed)
	device = utils.set_device(args)
	meddata = loader.MedicalDATA(args=args, device=device)
	model = models.DrugResponse(data=meddata, args=args, device=device)


if __name__ == '__main__':
	run(parser())
	print()
