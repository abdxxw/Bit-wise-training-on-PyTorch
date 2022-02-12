
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class LinearBit(nn.Module):

    def __init__(self, input_dim, output_dim, config):

        super(LinearBit, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.standard_kernel = config["standard_kernel"]
        self.trainBits = config["trainableBits"]
        self.wbits = len(self.trainBits)
        self.bittensor = config["pretrained_bittensor"]
        self.inference_sequence = config["inference_sequence"]
        self.name = name

        # converts virtual bits to binary coefficients
        self.tobit = to_bit
        self.tosign = to_sign

        self.kshape = (self.input_dim, self.output_dim)

        krnl_shape = (self.input_dim, self.output_dim)
        krnl_shape_bitwise = (self.wbits,) + krnl_shape
        print("Building Layer", self.name, krnl_shape)

        self.desired_std = np.sqrt(2 / np.prod(krnl_shape[:-1]))
        signbit = self.inference_sequence[1]
        magnitudebits = range(self.inference_sequence[0], self.inference_sequence[1])

        if self.standard_kernel == False:

            if self.wbits > 1:
                predefined_bit_tensor = predefine_nonzeroweight_bittensor((self.wbits,) + krnl_shape)
            else:
                print("number of bits must be greater than 1")

            self.magnitude_block = []
            for i in magnitudebits:
                self.magnitude_block.append(nn.Parameter(torch.tensor(predefined_bit_tensor[i, ...]).to(device),
                                                         requires_grad=bool(self.trainBits[i])))
            self.magnitude_block = nn.ParameterList(self.magnitude_block)

            if len(self.bittensor) > 0:
                self.sign_bit = nn.Parameter(torch.tensor(predefined_bit_tensor[signbit, ...]).to(device),
                                             requires_grad=bool(self.trainBits[signbit]))
            else:
                self.sign_bit = nn.Parameter(torch.randn(krnl_shape).to(device),
                                             requires_grad=bool(self.trainBits[signbit]))
                nn.init.kaiming_normal_(self.sign_bit, mode='fan_in')



        else:
            # uniform distribution with Kaiming He initialization technique
            self.weight = nn.Parameter(torch.randn(krnl_shape))
            nn.init.kaiming_normal_(self.weight, mode='fan_in')

    def forward(self, x):
        return F.linear(x, self.get_weight().T)

    def get_weight(self):
        if self.standard_kernel:
            return self.weight
        else:
            self.weight = calculate_number(self.tosign, self.tobit, self.magnitude_block, self.sign_bit)
            self.alpha = calc_scaling_factor(self.weight.clone().detach().cpu().numpy(),
                                             self.desired_std)  # for good convergence
            self.weight *= self.alpha
            return self.weight.float().to(device)

    def get_bits(self):
        bittensor = []
        for i in range(len(self.magnitude_block)):
            bittensor.append(self.magnitude_block[i])
        bittensor.append(self.sign_bit)

        return bittensor

    def get_nzp(self):
        return get_weight_types(self.get_weight())


class Conv2dBit(nn.Module):

    def __init__(self, input_dim, filters, kernel_size, stride, padding, config):

        super(Conv2dBit, self).__init__()

        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_dim = input_dim
        self.standard_kernel = config["standard_kernel"]
        self.trainBits = config["trainableBits"]
        self.wbits = len(self.trainBits)
        self.bittensor = config["pretrained_bittensor"]
        self.inference_sequence = config["inference_sequence"]
        self.name = name
        # converts virtual bits to binary coefficients
        self.tobit = to_bit
        self.tosign = to_sign


        krnl_shape = list((self.kernel_size, self.kernel_size)) + [self.input_dim, self.filters]

        krnl_shape_bitwise = [self.wbits]
        krnl_shape_bitwise.extend(krnl_shape)

        print("Building Layer", self.name, krnl_shape)

        self.desired_std = np.sqrt(2 / np.prod(krnl_shape[:-1]))
        signbit = self.inference_sequence[1]

        magnitudebits = range(self.inference_sequence[0], self.inference_sequence[1])

        if self.standard_kernel == False:

            if self.wbits > 1:
                predefined_bit_tensor = predefine_nonzeroweight_bittensor(krnl_shape_bitwise)
            else:
                print("number of bits must be greater than 1")

            self.magnitude_block = []
            for i in magnitudebits:
                self.magnitude_block.append(nn.Parameter(torch.tensor(predefined_bit_tensor[i, ...]).to(device),
                                                         requires_grad=bool(self.trainBits[i])))
            self.magnitude_block = nn.ParameterList(self.magnitude_block)
            if len(self.bittensor) > 0:
                self.sign_bit = nn.Parameter(torch.tensor(predefined_bit_tensor[signbit, ...]).to(device),
                                             requires_grad=bool(self.trainBits[signbit]))
            else:
                self.sign_bit = nn.Parameter(torch.randn(krnl_shape).to(device),
                                             requires_grad=bool(self.trainBits[signbit]))
                nn.init.kaiming_normal_(self.sign_bit, mode='fan_in')



        else:
            # uniform distribution with Kaiming He initialization technique
            self.kernel = nn.Parameter(torch.randn(krnl_shape))
            nn.init.kaiming_normal_(self.kernel, mode='fan_in')

    def forward(self, x):
        return F.conv2d(x, self.get_kernel().T, stride=self.stride, padding=self.padding)

    def get_kernel(self):
        if self.standard_kernel:
            return self.kernel
        else:
            self.kernel = calculate_number(self.tosign, self.tobit, self.magnitude_block, self.sign_bit)
            self.alpha = calc_scaling_factor(self.kernel.clone().detach().cpu().numpy(),
                                             self.desired_std)  # for good convergence
            self.kernel *= self.alpha
            return self.kernel.float().to(device)

    def get_bits(self):
        bittensor = []
        for i in range(len(self.magnitude_block)):
            bittensor.append(self.magnitude_block[i])
        bittensor.append(self.sign_bit)

        return bittensor

    def get_nzp(self):
        return get_weight_types(self.get_kernel())