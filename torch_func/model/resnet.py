import torch
import torch.nn as nn
from .utils import eager_quantize_model,fx_quantize_model
from .conv import DirectConv2d, PadConv2d, fuse_module, AlignW
import direct_conv2d 

def crop_out(out,margin=1):
    size2  = out.size(2)
    return out[:,:,margin:size2-margin,:]

class AlignAddLayer(nn.Module):
    def forward(self, x,y):
        # 计算x和y的第3和4个维度的size，并且对大的采用中心裁剪的方式对齐到小的那个Tensor
        x_size = x.size()
        y_size = y.size()
        size = [min(x_size[i],y_size[i]) for i in range(4)]
        x_margin_2 = max((x_size[2]-size[2])//2,0)
        x_margin_3 = max((x_size[3]-size[3])//2,0)
        y_margin_2 = max((y_size[2]-size[2])//2,0)
        y_margin_3 = max((y_size[3]-size[3])//2,0)
        x = x[:,:,x_margin_2:x_margin_2+size[2],x_margin_3:x_margin_3+size[3]]
        y = y[:,:,y_margin_2:y_margin_2+size[2],y_margin_3:y_margin_3+size[3]]
        # x = x[:size[0],:size[1],:size[2],:size[3]]
        # y = y[:size[0],:size[1],:size[2],:size[3]]
        return x+y


class FloatBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(FloatBasicBlock, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.relu = nn.ReLU(inplace=True)
        self.align_width = 12 # 6
        self.margin = 0
        # if in_channels==3:
        #     self.alignw = AlignW(self.align_width,True)
        # else:
        # self.alignw = AlignW(self.align_width,False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if stride == 1:
            self.margin = 1
            self.conv1 = nn.Sequential(
                AlignW(self.align_width,False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            AlignW(self.align_width,False),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
        self.alignadd = AlignAddLayer()

    def forward(self, x):
        # out = self.quant(x)
        # print(f"x shape: {x.shape}")
        out = self.conv1(x)
        out = crop_out(out,self.margin)
        # print(f"out shape: {out.shape}, x shape: {x.shape}")
        out = self.conv2(out)
        out = crop_out(out)
        # print(f"out shape: {out.shape}, x shape: {x.shape}")
        x = self.shortcut(x)
        out = self.dequant(out)
        x = self.dequant(x)
        # print(f"out shape: {out.shape}, x shape: {x.shape}")
        # out += self.shortcut(x)
        out = self.alignadd(out,x)
        out = self.relu(out)
        out = self.quant(out)
        return out
    


class DC_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 W_bits, A_bits, MT=True, stride=1, 
                 show_time=False):
        super(DC_layer, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.MT = MT
        self.stride = stride
        # self.dilation = dilation
        # self.show_time = show_time
        self.show_time = False
        self.depth_conv = False
        max_w = 2**(W_bits) - 1
        self.weight = torch.randint(0, max_w, 
                               (self.out_channels, 
                                self.in_channels, 
                                self.kernel_size,
                                self.kernel_size)).int()

    def forward(self, x):
        x = x.int()
        # if self.show_time:
        #     print(f"x shape {x.shape}, weight shape {self.weight.shape}")
        x =  direct_conv2d.direct_conv2d(x, self.weight, self.W_bits, self.A_bits, self.MT,self.stride,self.show_time,self.depth_conv)
        return x


class DCBasicBlock(nn.Module):
    expansion = 1
    __WABits2Imgsize__ = {(5,5):112,(4,4):112,(3,3):112,(2,2):112,(1,1):112}
    def __init__(self, in_channels, out_channels, stride=1, W_bits=3, A_bits=3, input_size=224):
        super(DCBasicBlock, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.relu = nn.ReLU(inplace=True)
        self.align_width = 12 # 6
        self.margin = 0
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.input_size = input_size
        self.max_img_size = self.__WABits2Imgsize__[(self.W_bits,self.A_bits)]
        # if in_channels==3:
        #     self.alignw = AlignW(self.align_width,True)
        # else:
        # self.alignw = AlignW(self.align_width,False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if stride == 1:
            self.margin = 1
            if input_size > self.max_img_size:
                self.conv1 = nn.Sequential(
                    AlignW(self.align_width,False),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=2, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.conv1 = nn.Sequential(
                    self.dequant,
                    AlignW(self.align_width,False),
                    DC_layer(in_channels, out_channels, kernel_size=3,W_bits=self.W_bits, A_bits=self.A_bits, MT=True),
                )

        if input_size > self.max_img_size:
            self.conv2 = nn.Sequential(
                AlignW(self.align_width,False),
                nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=2, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
                self.dequant
            )
        elif stride == 1:
            self.conv2 = nn.Sequential(
                AlignW(self.align_width,False),
                DC_layer(out_channels, out_channels * self.expansion, kernel_size=3, W_bits=self.W_bits, A_bits=self.A_bits, MT=True)
            )
        else:
            self.conv2 = nn.Sequential(
                self.dequant,
                AlignW(self.align_width,False),
                DC_layer(out_channels, out_channels * self.expansion, kernel_size=3, W_bits=self.W_bits, A_bits=self.A_bits, MT=True)
            )

        self.shortcut = nn.Sequential(self.dequant)
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
                self.dequant
            )
            # if stride == 1:
            #     self.shortcut = nn.Sequential(
            #         AlignW(self.align_width,False),
            #         nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=1, bias=False),
            #         nn.BatchNorm2d(self.expansion * out_channels)
            #     )
        self.alignadd = AlignAddLayer()

    def forward(self, x):
        # out = self.quant(x)
        # print(f"x shape: {x.shape} {x.dtype} {self.shortcut}")
        out = self.conv1(x)
        out = crop_out(out,self.margin)
        # print(f"out shape: {out.shape} {out.dtype}, x shape: {x.shape} {x.dtype}")
        out = self.conv2(out)
        out = crop_out(out)
        # print(f"out shape: {out.shape} {out.dtype}, x shape: {x.shape} {x.dtype}")
        x = self.shortcut(x)
        # out = self.dequant(out)
        # x = self.dequant(x)
        # print(f"out shape: {out.shape} {out.dtype}, x shape: {x.shape} {x.dtype}")
        # out += self.shortcut(x)
        out = self.alignadd(out,x)
        out = self.relu(out)
        out = self.quant(out)
        return out

class Int8BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(FloatBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet_DC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000,W_bits=5,A_bits=5,init_input=224):
        super(ResNet_DC, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.in_channels = 64
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.init_input = init_input
        self.max_img_input = init_input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        self.max_img_input = self.max_img_input//stride
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, W_bits=self.W_bits, A_bits=self.A_bits, input_size=self.max_img_input))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.relu = nn.ReLU(inplace=True)
        self.align_width = 12 # 6
        self.margin = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if stride == 1:
            self.conv2 = nn.Sequential(
                AlignW(self.align_width,False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.alignadd = AlignAddLayer()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x = self.shortcut(x)
        out = self.dequant(out)
        x = self.dequant(x)
        out = self.alignadd(out,x)
        # out += self.shortcut(x)
        out = self.relu(out)
        out = self.quant(out)
        return out

class DCBottleneck(nn.Module):
    expansion = 4
    __WABits2Imgsize__ = {(5,5):112,(4,4):112,(3,3):112,(2,2):112,(1,1):112}
    def __init__(self, in_channels, out_channels, stride=1,W_bits=3, A_bits=3, input_size=224):
        super(DCBottleneck, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.align_width = 12 # 6
        self.margin = 0
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.input_size = input_size
        self.max_img_size = self.__WABits2Imgsize__[(self.W_bits,self.A_bits)]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if stride == 1:
            if input_size > self.max_img_size:
                self.conv2 = nn.Sequential(
                    AlignW(self.align_width,False),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=2, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                self.conv2 = nn.Sequential(
                    self.dequant,
                    AlignW(self.align_width,False),
                    DC_layer(out_channels, out_channels, kernel_size=3, W_bits=self.W_bits, A_bits=self.A_bits, MT=True),
                    self.quant
                )

        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.alignadd = AlignAddLayer()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        x = self.shortcut(x)
        out = self.dequant(out)
        x = self.dequant(x)
        out = self.alignadd(out,x)
        # out += self.shortcut(x)
        out = self.relu(out)
        out = self.quant(out)
        return out

def resnet18():
    return ResNet(FloatBasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(FloatBasicBlock, [3, 4, 6, 3])

def resnet_qint8(block,num_blocks, shape=(1,3,224,224),engine_name = 'qnnpack'):
    model = ResNet(block, num_blocks)
    # model = set_model_quant_bits(model, 8, 8)
    # fuse model layers
    model = fuse_module(model, inplace=True)
    # Eager quantization
    model = eager_quantize_model(model,shape,engine_name=engine_name,inplace=True,prepare=True)
    return model

def resnet18_qint8(shape=(1,3,224,224),engine_name = 'qnnpack'):
    return resnet_qint8(FloatBasicBlock, [2, 2, 2, 2], shape, engine_name)
    # model = ResNet(FloatBasicBlock, [2, 2, 2, 2])
    # # model = set_model_quant_bits(model, 8, 8)
    # # fuse model layers
    # model = fuse_module(model, inplace=True)
    # # Eager quantization
    # model = eager_quantize_model(model,shape,engine_name=engine_name,inplace=True,prepare=True)
    # return model

def resnet34_qint8(shape=(1,3,224,224),engine_name = 'qnnpack'):
    return resnet_qint8(FloatBasicBlock, [3, 4, 6, 3], shape, engine_name)

def resnet_dc(block,num_blocks,W_bits=5, A_bits=5,shape = (1,3,224,224),engine_name="qnnpack"):
    model = ResNet_DC(block, num_blocks, 1000, W_bits=W_bits, A_bits=A_bits, init_input=56)
    model = fuse_module(model, inplace=True)
    model = eager_quantize_model(model,shape,False,engine_name,inplace=True)
    return model

def resnet18_dc(W_bits, A_bits):
    return resnet_dc(DCBasicBlock, [2, 2, 2, 2], W_bits=W_bits, A_bits=A_bits)
    # model = ResNet_DC(DCBasicBlock, [2, 2, 2, 2], 1000, W_bits=W_bits, A_bits=A_bits, init_input=224)
    # model = fuse_module(model, inplace=True)
    # model = eager_quantize_model(model,shape,False,engine_name,inplace=True)
    # return model

def resnet34_dc(W_bits, A_bits):
    return resnet_dc(DCBasicBlock, [3, 4, 6, 3], W_bits=W_bits, A_bits=A_bits)


# def resnet18_qint8(shape=(1,3,224,224),engine_name = 'qnnpack'):
#     model = ResNet(FloatBasicBlock, [2, 2, 2, 2])
#     # model = set_model_quant_bits(model, 8, 8)
#     # fuse model layers
#     model = fuse_module(model, inplace=True)
#     # Eager quantization
#     model = eager_quantize_model(model,shape,engine_name=engine_name,inplace=True,prepare=True)
#     return model

# def ResNet34():
#     return ResNet(FloatBasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def resnet50_qint8(shape=(1,3,224,224),engine_name = 'qnnpack'):
    return resnet_qint8(Bottleneck, [3, 4, 6, 3], shape, engine_name)

def resnet101_qint8(shape=(1,3,224,224),engine_name = 'qnnpack'):
    return resnet_qint8(Bottleneck, [3, 4, 23, 3], shape, engine_name)

def resnet50_dc(W_bits, A_bits):
    return resnet_dc(DCBottleneck, [3, 4, 6, 3], W_bits=W_bits, A_bits=A_bits)

def resnet101_dc(W_bits, A_bits):
    return resnet_dc(DCBottleneck, [3, 4, 23, 3], W_bits=W_bits, A_bits=A_bits)