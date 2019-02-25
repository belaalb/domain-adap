import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.autograd import Variable
from tqdm import tqdm
from test import test


class TrainLoop(object):

	def __init__(self, model, optimizer, source_loader, target_loader, checkpoint_path=None, checkpoint_epoch=None, cuda=True, target_name='mnist_m'):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch = os.path.join(self.checkpoint_path, 'cp_{}ep.pt')

		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_adap)
		self.lambda_adap = 1. / (1. + 10 * p) ** 0.75
		self.source_loader = source_loader
		self.target_loader = target_loader
		self.history = {'loss': []}
		self.cur_epoch = 0
		self.target_name = target_name

	def train(self, n_epochs=1, save_every=1):

		while self.cur_epoch < n_epochs:

			len_dataloader = min(len(self.source_loader), len(self.target_loader))

			source_iter = iter(self.source_loader)
			target_iter = iter(self.target_loader)

			print('Epoch {}/{}'.format(self.cur_epoch + 1, n_epochs))

			i = 0
			cur_loss = 0
			while i < len_dataloader:
				p = float(i + self.cur_epoch * len_dataloader) / (n_epochs*len_dataloader)
				self.lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

				try:
					batch_source = source_iter.next()
					batch_target = target_iter.next()
				except:
					source_iter = iter(self.source_loader)
					target_iter = iter(self.target_loader)
					batch_source = source_iter.next()
					batch_target = target_iter.next()

				cur_loss += self.train_step(batch_source, batch_target)
				i += 1

			if self.cur_epoch % save_every == 0:
				self.checkpointing()

			self.history['loss'].append(cur_loss/i)

			print('Current loss: {}.'.format(cur_loss/i))

			if self.cur_epoch % save_every == 0:
				test(self.target_name, self.cur_epoch, self.checkpoint_path, self.cuda_mode)

			self.cur_epoch += 1

		# saving final models
		print('Saving final model...')
		self.checkpointing()


	def train_step(self, batch_source, batch_target):
		self.model.train()

		x_source, y_source = batch_source
		source_labels = torch.zeros(y_source.size(), dtype=torch.long)

		x_target, _ = batch_target
		target_labels = torch.ones(x_target.size(0), dtype=torch.long)

		if self.cuda_mode:
			x_source = x_source.cuda()
			y_source = y_source.cuda()
			source_labels = source_labels.cuda() 

		class_out, domain_out = self.model.forward(x_source, self.lambda_)		
		loss_class = torch.nn.NLLLoss()(class_out, y_source)
		loss_source = torch.nn.NLLLoss()(domain_out, source_labels)
		
		if self.cuda_mode:
			x_target = x_target.cuda()
			target_labels = target_labels.cuda()

		_, domain_out = self.model.forward(x_target, self.lambda_)		
		loss_target = torch.nn.NLLLoss()(domain_out, target_labels)

		loss = loss_class + loss_source + loss_target 

		loss.backward()
		self.optimizer.step()

		#self.print_grad_norms()

		return loss.item() 


	def checkpointing(self):
		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
				'optimizer_state': self.optimizer.state_dict(),
				'history': self.history,
				'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch.format(self.cur_epoch))

	def load_checkpoint(self, epoch):
		ckpt = self.save_epoch.format(epoch)

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load scheduler state
			# Load history
			self.history = ckpt['history']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(filter(lambda p: p.grad is not None, self.model.parameters())):
			norm += params.grad.norm(2).item()
		print('Sum of grads norms: {}'.format(norm))



