import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.autograd import Variable
from tqdm import tqdm


class TrainLoop(object):

	def __init__(self, model, optimizer, source_loader, target_loader, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
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
		self.source_loader = source_loader
		self.target_loader = target_loader
		self.history = {'loss': []}
		self.total_iters = 0
		self.cur_epoch = 0

	def train(self, n_epochs=1, save_every=1):

		len_dataloader = min(len(source_loader), len(target_loader))
		source_iter = tqdm(iter(source_loader))
		target_iter = iter(target_loader)

		while self.cur_epoch < n_epochs:

			print('Epoch {}/{}'.format(self.cur_epoch + 1, n_epochs))

			i = 0
			cur_loss
			while i < len_dataloader:
				p = float(i + self.cur_epoch * len_dataloader) / n_epochs / len_dataloader
				self.alpha = 2. / (1. + np.exp(-10 * p)) - 1

				batch_source = source_iter.next()
				batch_target = target_iter.next()
				cur_loss += self.train_step(batch_source, batch_target)
				i += 1
			
			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0:
				self.checkpointing()

			self.history['loss'].append(cur_loss/i)

			print('Current loss: {}.'.format(cur_loss/i))

		# saving final models
		print('Saving final model...')
		self.checkpointing()


	def train_step(self, batch_source, batch_target):
		self.model.train()

		x_source, y_source = batch_source
		source_labels = torch.zeros(y_source.size())

		x_target, _ = batch_target
		target_labels = torch.zeros(y_target.size())

		if self.cuda_mode:
			x_source = x_source.cuda()
			y_source = y_source.cuda()
			source_labels = source_labels.cuda() 

		class_out, domain_out = self.model.forward(x_source, self.alpha)		
		loss_class = torch.nn.NLLLoss()(class_out, y_source)
		loss_source = torch.nn.NLLLoss()(domain_out, source_labels)
		
		if self.cuda_mode:
			x_target = x_target.cuda()
			target_labels = target_labels.cuda()

		_, domain_out = self.model.forward(x_target, self.alpha)		
		loss_target = torch.nn.NLLLoss()(domain_out, y_target)

		loss = loss_class + loss_source + loss_target 

		loss.backward()
		self.optimizer.step()

		return loss.item() 


	def checkpointing(self):
		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
				'optimizer_state': self.optimizer.state_dict(),
				'scheduler_state': self.scheduler.state_dict(),
				'history': self.history,
				'total_iters': self.total_iters,
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
			self.scheduler.load_state_dict(ckpt['scheduler_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']

		else:
			print('No checkpoint found at: {}'.format(ckpt))



