import torch
import torch.nn as nn
import torch.nn.init as init


class ISP(nn.Module):

	def __init__(self):
		super(ISP, self).__init__()

		self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
		self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2)

		self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2)

		self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.upv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
		self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

		self.upv5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
		self.conv5_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
		self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

		self.conv6_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

	def forward(self, x):
		conv1 = self.lrelu(self.conv1_1(x))
		conv1 = self.lrelu(self.conv1_2(conv1))
		pool1 = self.pool1(conv1)

		conv2 = self.lrelu(self.conv2_1(pool1))
		conv2 = self.lrelu(self.conv2_2(conv2))
		pool2 = self.pool1(conv2)

		conv3 = self.lrelu(self.conv3_1(pool2))
		conv3 = self.lrelu(self.conv3_2(conv3))

		up4 = self.upv4(conv3)
		up4 = torch.cat([up4, conv2], 1)
		conv4 = self.lrelu(self.conv4_1(up4))
		conv4 = self.lrelu(self.conv4_2(conv4))

		up5 = self.upv5(conv4)
		up5 = torch.cat([up5, conv1], 1)
		conv5 = self.lrelu(self.conv5_1(up5))
		conv5 = self.lrelu(self.conv5_2(conv5))

		conv6 = self.conv6_1(conv5)
		out = nn.functional.pixel_shuffle(conv6, 2)
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0.0, 0.02)
				if m.bias is not None:
					m.bias.data.normal_(0.0, 0.02)
			if isinstance(m, nn.ConvTranspose2d):
				m.weight.data.normal_(0.0, 0.02)

	def lrelu(self, x):
		outt = torch.max(0.2 * x, x)
		return outt
