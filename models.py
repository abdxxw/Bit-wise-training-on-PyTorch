
from layers import *




class ImageClassificationBase(nn.Module):
    '''
    since we are on the image classification task we use this general class for training and validation steps
    '''
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


###################################################################################################
###################################################################################################

                                 #LeNet for MNIST dataset

###################################################################################################
###################################################################################################


class LeNet(ImageClassificationBase):
    def __init__(self,config):
        super(LeNet, self).__init__()
        self.conv1 = Conv2dBit(1, 6, kernel_size=5, stride=1, padding=0, config=config)
        self.conv2 = Conv2dBit(6, 16, kernel_size=5, stride=1, padding=0, config=config)
        self.fc1   = LinearBit(16*4*4, 120,config=config)
        self.fc2   = LinearBit(120, 84,config=config)
        self.fc3   = LinearBit(84, 10,config=config)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


###################################################################################################
###################################################################################################

                                    # ResNet for CIFAR dataset

###################################################################################################
###################################################################################################

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, config, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dBit(in_planes, planes,kernel_size=3,padding=1,stride=stride,config=config)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dBit(planes, planes,kernel_size=3,padding=1,stride=1,config=config)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2dBit(in_planes, self.expansion*planes,kernel_size=1, stride=stride, padding=0,config=config),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ImageClassificationBase):
    def __init__(self, block, num_blocks, config, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2dBit(3, 16, kernel_size=3,padding=1,stride=1,config=config)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], config, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], config, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], config, stride=2)
        self.linear = LinearBit(64, num_classes,config=config)


    def _make_layer(self, block, planes, num_blocks, config, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, config, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(config):
    return ResNet(BasicBlock, [3, 3, 3],config)


###################################################################################################
###################################################################################################

                                    # Conv6 for CIFAR dataset

###################################################################################################
###################################################################################################


class Conv6(ImageClassificationBase):

    def __init__(self,config):
        super(Conv6, self).__init__()

        self.layer1 = torch.nn.Sequential(
            Conv2dBit(3, 64, kernel_size=3, stride=1, padding=1,config=config),
            torch.nn.ReLU(),
            Conv2dBit(64, 64, kernel_size=3, stride=1, padding=1,config=config),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))

        self.layer2 = torch.nn.Sequential(
            Conv2dBit(64, 128, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            Conv2dBit(128, 128, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))

        self.layer3 = torch.nn.Sequential(
            Conv2dBit(128, 256, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            Conv2dBit(256, 256, kernel_size=3, stride=1, padding=1, config=config),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = LinearBit(5 * 5 * 256, 256, config)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        self.fc2 = LinearBit(256, 256, config)
        self.layer5 = torch.nn.Sequential(
            self.fc2,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc3 = LinearBit(256, 10, config)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc3(out)
        return out
