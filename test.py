import os
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from data_loader import Loader
from torchvision import datasets


def test(dataset_name, epoch, checkpoint_path, cuda):
	assert dataset_name in ['mnist', 'mnist_m']
	image_root = os.path.join('.', 'data', dataset_name)

	batch_size = 128
	image_size = 28
	alpha = 0

	"""load data"""

	if dataset_name == 'mnist_m':
		test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
		img_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
		dataset = Loader(data_root=os.path.join(image_root, 'mnist_m_test'),data_list=test_list, transform=img_transform)
	else:
		img_transform_mnist = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
		dataset = datasets.MNIST(root=image_root, train=False, transform=img_transform_mnist)
		dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)


	model = torch.load(os.path.join(checkpoint_path, 'cp_{}ep'.format(epoch)) + '.pt'))
	model = model.eval()

	if cuda:
	  model = model.cuda()

	len_dataloader = len(dataloader)
	data_target_iter = iter(dataloader)

	i = 0
	n_total = 0
	n_correct = 0

	while i < len_dataloader:

		# test model using target data
		x, y = data_target_iter.next()

		if cuda:
			x = x.cuda()
			y = y.cuda()

		class_output, _ = model(input_data=x, alpha=alpha)
		pred = class_output.data.max(1, keepdim=True)[1]
		n_correct += pred.eq(y.data.view_as(pred)).cpu().sum()
		n_total += batch_size

		i += 1

    accu = n_correct * 1.0 / n_total

    print('Epoch:{}, accuracy of the {}, dataset: {}.'.format(epoch, dataset_name, accu))
