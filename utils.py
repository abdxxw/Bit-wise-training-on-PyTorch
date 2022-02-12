

class to_bit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(nn.ReLU()(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output


class to_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (-1) ** torch.sign(nn.ReLU()(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return -grad_output



def predefine_nonzeroweight_bittensor(shape):
    nlp = np.prod(shape[:-1])
    a = np.sqrt(2 / nlp)
    nbits = shape[0]
    distribution = a * np.random.normal(0, 1, shape)
    # mlp here
    if len(shape) == 3:
        for i in range(shape[1]):
            for j in range(shape[2]):
                while np.all(distribution[:-1, i, j] <= 0):
                    distribution[:-1, i, j] = a * np.random.normal(0, 1, nbits - 1)

    # kernel here
    if len(shape) == 5:
        for in_channels in range(shape[3]):
            for out_channels in range(shape[4]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        while np.all(distribution[:-1, i, j, in_channels, out_channels] <= 0):
                            distribution[:-1, i, j, in_channels, out_channels] = a * np.random.normal(0, 1, nbits - 1)

    return distribution


def calc_scaling_factor(k, target):
    current_std = np.std(k)

    if current_std == 0:
        print("something's wrong, the standard deviation is zero!")
        return 1

    ampl = 1
    eps = 0.001
    min = 0
    max = ampl

    steps = 0
    while np.abs(current_std - target) / target > eps:
        qk = k * ampl
        current_std = np.std(qk)

        if current_std > target:
            max = ampl
            ampl = (max + min) / 2
        elif current_std < target:
            min = ampl
            ampl = (max + min) / 2
        steps += 1

    return ampl


def calculate_number(signfunction, maskfunction, magnitude_block, sign_slice):
    if len(magnitude_block) == 0:
        magnitude = 1
    else:

        magnitude = 0
        for i in range(len(magnitude_block)):
            magnitude += maskfunction.apply(magnitude_block[i]) * (2 ** i)
    # make kernel
    kernel = signfunction.apply(sign_slice) * magnitude
    return kernel


def get_weight_types(k):
    """
    returns the number of negative, zero and positive weights
    """
    neg = np.count_nonzero(k < 0)
    zeros = np.count_nonzero(k == 0)
    pos = np.count_nonzero(k > 0)

    return neg, zeros, pos


def getNZP(net):
    nsum = 0
    zsum = 0
    psum = 0

    for l in net.modules():

        if isinstance(l, Conv2dBit) or isinstance(l, LinearBit):
            neg, zero, pos = l.get_nzp()
            nsum += neg
            zsum += zero
            psum += pos

    return nsum, zsum, psum