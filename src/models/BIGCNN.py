import torch


# class BIGCNN(torch.nn.Module):  #CCN FROM https://medium.com/@alitbk/image-classification-in-a-nutshell-5-different-modelling-approaches-in-pytorch-with-cifar100-8f690866b373
#     def __init__(self, in_channels, hidden_size, num_classes):
#         super(BIGCNN, self).__init__()
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_size
#         self.num_classes = num_classes
#         self.activation = torch.nn.ReLU(True)
#
#         self.features = torch.nn.Sequential(
#             # input: 3 x 32 x 32
#             torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             # output: 32 x 32 x 32
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             # output: 64 x 32 x 32
#             torch.nn.ReLU(),
#             # output: 64 x 32 x 32
#             torch.nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
#
#             torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
#
#             torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
#         )
#         self.classifier = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.Linear(256*4*4, 1024),
#             torch.nn.ReLU(),
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 100)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


def conv_block(in_channels, out_channels, pool=False):
    layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              torch.nn.BatchNorm2d(out_channels),
              torch.nn.ReLU(inplace=True)]
    if pool: layers.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*layers)

class BIGCNN(torch.nn.Module):  #RESNET9
    def __init__(self, in_channels, hidden_size, num_classes):
        super(BIGCNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_size
        self.num_classes = num_classes
        self.activation = torch.nn.ReLU(True)

        self.conv1 = conv_block(self.in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = torch.nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = torch.nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = torch.nn.Sequential(torch.nn.MaxPool2d(4),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out