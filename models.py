import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class CNNModel(nn.Module):

	def __init__(self):
		super(CNNModel, self).__init__()
		self.feature = nn.Sequential()
		self.feature.add_module('f_conv1', nn.Conv2d(3, 32, kernel_size=5, padding=2))
		self.feature.add_module('f_pool1', nn.MaxPool2d(2))
		self.feature.add_module('f_relu1', nn.ReLU())
		self.feature.add_module('f_conv2', nn.Conv2d(32, 48, kernel_size=5, padding=2))
		self.feature.add_module('f_pool2', nn.MaxPool2d(2))
		self.feature.add_module('f_relu2', nn.ReLU())

		self.class_classifier = nn.Sequential()
		self.class_classifier.add_module('c_fc1', nn.Linear(48 * 7 * 7, 100))
		self.class_classifier.add_module('c_relu1', nn.ReLU())
		self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
		self.class_classifier.add_module('c_relu2', nn.ReLU())
		self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
		#self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

		self.domain_classifier = nn.Sequential()
		self.domain_classifier.add_module('d_fc1', nn.Linear(48 * 7 * 7, 100))
		self.domain_classifier.add_module('d_relu1', nn.ReLU())
		self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
		#self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

	def forward(self, input_data, alpha):
		feature = self.feature(input_data)
		feature = feature.view(-1, 48 * 7 * 7)
		reverse_feature = ReverseLayer.apply(feature, alpha)
		class_output = F.softmax(self.class_classifier(feature))
		domain_output = F.softmax(self.domain_classifier(reverse_feature))

		return class_output, domain_output
