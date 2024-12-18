import torch
import torch.nn as nn
from .utils import eager_quantize_model,fx_quantize_model
from .conv import DirectConv2d, PadConv2d, fuse_module, AlignW
import direct_conv2d 

class VGG_float(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super().__init__()
        self.cfg = cfg
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.align_width = 6
        self.features = self.make_layers(self.cfg, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.cfg[-1], num_classes),
        )
        # self.alignw = AlignW(12)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in self.cfg:
            if in_channels== 3:
                alignw = AlignW(self.align_width,True)
            else:
                alignw = AlignW(self.align_width,False)
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2)
                if batch_norm:
                    layers += [alignw, conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [alignw, conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cfgs = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

def vgg16_float():
    return VGG_float(cfgs['VGG16'], batch_norm=True)

class VGG_int8(VGG_float):
    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
        

def vgg16_qint8(name = "VGG16",shape=(1,3,224,224),engine_name = 'qnnpack'):
    model = VGG_int8(cfgs[name], batch_norm=True)
    # model = set_model_quant_bits(model, 8, 8)
    # fuse model layers
    model = fuse_module(model, inplace=True)
    # Eager quantization
    model = eager_quantize_model(model,shape,engine_name=engine_name,inplace=True,prepare=True)
    return model

class DC_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, W_bits, A_bits, MT=True, stride=1, show_time=False):
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
        self.show_time = show_time
        self.depth_conv = False
        # self.show_time = False
        max_w = 2**(W_bits) - 1
        self.weight = torch.randint(0, max_w, 
                               (self.out_channels, 
                                self.in_channels, 
                                self.kernel_size,
                                self.kernel_size)).int()

    def forward(self, x):
        x = x.int()
        x =  direct_conv2d.direct_conv2d(x, self.weight, self.W_bits, self.A_bits, self.MT,self.stride,self.show_time,self.depth_conv)
        return x

class VGG_DC(nn.Module):
    __WABits2Imgsize__ = {(5,5):28,(4,4):56,(3,3):56,(2,2):112,(1,1):112}
    def __init__(self, cfg, 
                 W_bits=5,A_bits=5,init_input=224,
                 batch_norm=False, num_classes=1000):
        super().__init__()
        self.cfg = cfg
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.init_input = init_input
        self.max_img_size = self.__WABits2Imgsize__[(self.W_bits,self.A_bits)]
        self.align_width = 6
        # if (self.W_bits+self.A_bits)<=4:
        #     self.align_width = 12
        self.features = self.make_layers(self.cfg, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            torch.ao.quantization.QuantStub(),
            nn.Linear(self.cfg[-1], num_classes),
        )
        # self.alignw = AlignW(12)
    
    def set_WA_bits(self,W_bits, A_bits):
        self.W_bits = W_bits
        self.A_bits = A_bits
        for n,m in self.features.named_modules():
            if isinstance(m, DC_layer):
                m.W_bits = W_bits
                m.A_bits = A_bits

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        input_size = self.init_input
        after_quant = False
        for v in self.cfg:
            if in_channels== 3:
                alignw = AlignW(self.align_width,True)
            else:
                alignw = AlignW(self.align_width,False)
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                input_size = input_size//2
            else:
                if input_size > self.max_img_size or in_channels==3:
                    if not after_quant:
                        layers+=[self.quant]
                        after_quant = True
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2)
                    if batch_norm:
                        layers += [alignw, conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [alignw, conv2d, nn.ReLU(inplace=True)]
                    
                else:
                    conv2d = DC_layer(in_channels, v, kernel_size=3, W_bits=self.W_bits, A_bits=self.A_bits, MT=True) 
                    if after_quant:
                        layers += [self.dequant]
                        after_quant = False
                    layers += [alignw,conv2d]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # x = self.quant(x)
        x = self.classifier(x)
        return x
    
def vgg16_dc(W_bits, A_bits,shape = (1,3,224,224),engine_name="qnnpack"):
    model = VGG_DC(cfgs['VGG16'], batch_norm=True, W_bits=W_bits, A_bits=A_bits)
    model = fuse_module(model, inplace=True)
    model = eager_quantize_model(model,shape,False,engine_name,inplace=True)
    return model



# Test Codes:
def Test():
    from HIPACK.torch_func.models.vggnet import vgg16_dc
    from HIPACK.torch_func.models.vggnet import vgg16_float, vgg16_qint8
    from models.utils import MeasureExecutionTime


    model = vgg16_dc(5,5)

    x= torch.rand(16,3,224,224)
    with MeasureExecutionTime():
        out = model(x)

    fmodel = vgg16_float()
    qmodel = vgg16_qint8()
    # with MeasureExecutionTime():
    #     out = fmodel(x) # 这个很慢
        
    with MeasureExecutionTime():
        out = qmodel(x)