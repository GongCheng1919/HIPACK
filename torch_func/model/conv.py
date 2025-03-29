import torch
import direct_conv2d
import torch
import torch.nn as nn
import direct_conv2d  # 确保已经导入direct_conv2d模块
import copy

from .utils import MeasureExecutionTime

def direct_conv2d_func(inp, weight, W_bits, A_bits, MT, padding, stride, dilation):
    return direct_conv2d.direct_conv2d(inp, weight, W_bits, A_bits, MT, padding, stride, dilation)

def direct_conv2d_func(inp, weight, W_bits, A_bits, MT):
    return direct_conv2d.direct_conv2d(inp, weight, W_bits, A_bits, MT)

class AlignW(nn.Module):
    def __init__(self,align_width=6,upsample=True):
        super(AlignW, self).__init__()
        self.align_width=align_width
        self.upsample  = upsample
    def forward(self,inp):
        W = inp.size(3)
        if self.upsample:
            W = int((W//self.align_width)*self.align_width+((W%self.align_width)>0)*self.align_width) 
        else:
            W = int((W//self.align_width)*self.align_width)
        W = max(W,self.align_width)
        # employ the new W to padding inp (with zero)
        if W>inp.size(3):
            inp = torch.nn.functional.pad(inp, (0, W-inp.size(3), 0, 0), mode='constant', value=0)
        else:
            step = (W%self.align_width)//2
            inp = inp[:,:,:,step:W+step]
        return inp

class DirectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, W_bits, A_bits, 
                 MT=True, padding=0, 
                 stride=1, dilation=1,
                 prepare_func=lambda x: x.int(),
                 post_func = lambda x: x,
                 measure_time=False,
                 keep_same_output_size=False,
                 align_w = False):
        super(DirectConv2d, self).__init__()
        self.prepare_func = prepare_func
        self.post_func = post_func
        self.measure_time = measure_time
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.W_bits = W_bits
        self.A_bits = A_bits
        self.MT = MT
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        max_w = 2**(W_bits) - 1
        max_a = 2**(A_bits) - 1
        regA = 2
        TN =2
        if (W_bits+A_bits)<3:
            TN = 4
        self.align_num = regA*self.kernel_size*TN
        self.weight = torch.randint(0, max_w, 
                               (self.out_channels, 
                                self.in_channels, 
                                self.kernel_size,
                                self.kernel_size)).int()
        self.keep_same_output_size = keep_same_output_size
        self.align_w = align_w
        if self.align_w:
            self.alignw = AlignW(self.align_num)
        # 初始化权重，这里假设权重是正方形的
        # 注意：权重的数据类型应该与期望的整型匹配
        # self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(torch.int32))
        # self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).int()
    def set_quant_bits(self, W_bits,A_bits):
        self.W_bits = W_bits
        self.A_bits = A_bits
        max_w = 2**(W_bits) - 1
        max_a = 2**(A_bits) - 1
        self.weight = torch.randint(0, max_w, 
                               (self.out_channels, 
                                self.in_channels, 
                                self.kernel_size,
                                self.kernel_size)).int()
        regA = 2
        TN =2
        if (W_bits+A_bits)<3:
            TN = 4
        self.align_num = self.kernel_size*TN
        self.alignw = AlignW(self.align_num)

    def evaluate_perf(self,output_shape,eval_time):
        self.flops = np.prod(output_shape)*np.prod(self.weight.shape)*2/self.in_channels
        perf = self.flops/eval_time
        # auto set 单位
        if perf>1e12:
            perf = perf/1e12
            unit = "TFLOPS"
        elif perf>1e9:
            perf = perf/1e9
            unit = "GFLOPS"
        elif perf>1e6:
            perf = perf/1e6
            unit = "MFLOPS"
        elif perf>1e3:
            perf = perf/1e3
            unit = "KFLOPS"
        return f"{perf:.2f}{unit}"

    def forward(self,inp):
        if self.measure_time:
            return self.measure_forward(inp)
        else:
            return self.direct_forward(inp)
    
    def direct_forward(self,inp):
        # 检查inp和weight的数据类型是否为整型
        if not inp.dtype == torch.int32:
            # raise TypeError("inp and weight must be of int32 type")
            inp = self.prepare_func(inp)
        # 检查inp和weight的形状是否匹配，不匹配则出错：
        if not inp.size(1) == self.in_channels:
            raise ValueError(f"input shape {inp.shape} does not match weight shape {self.weight.shape}")
        # 此处还需要对inp做padding用于计算
        if self.align_w:
            inp = self.alignw(inp)
        # W = inp.size(3)
        # W = int((W//self.align_num)*self.align_num+((W%self.align_num)>0)*self.align_num) 
        # # employ the new W to padding inp (with zero)
        # if W!=inp.size(3):
        #     inp = torch.nn.functional.pad(inp, (0, W-inp.size(3), 0, 0), mode='constant', value=0)
        # 调用direct_conv2d.direct_conv2d函数
        # print(inp.shape,self.weight.shape)
        # start = time.perf_counter()
        # direct_conv2d_func(inp, self.weight, self.W_bits, self.A_bits, self.MT)
        output =  direct_conv2d.direct_conv2d(inp, self.weight, self.W_bits, self.A_bits, self.MT)
        # eclapsed = time.perf_counter()-start
        # print(output.shape,f"time eclapsed: {eclapsed}s")
        # self.full_perf = self.evaluate_perf(output.shape,eclapsed)
        # 如果输出只要中间的一块，需要对output进行裁剪，其从0位置开始，裁剪出和inp一样大的区域
        if self.keep_same_output_size:
            output = output[:, :, 1:1+inp.size(2), 1:1+inp.size(2)]
        output = self.post_func(output)
        # print(output.shape)
        # self.cutted_perf = self.evaluate_perf(output.shape,eclapsed)
        return output
    
    def measure_forward(self, inp):
        with MeasureExecutionTime(measure_name="Check inp type and shape"):
            # 检查inp和weight的数据类型是否为整型
            if not inp.dtype == torch.int32:
                # raise TypeError("inp and weight must be of int32 type")
                inp = self.prepare_func(inp)
            # 检查inp和weight的形状是否匹配，不匹配则出错：
            if not inp.size(1) == self.in_channels:
                raise ValueError(f"input shape {inp.shape} does not match weight shape {self.weight.shape}")
        with MeasureExecutionTime(measure_name="Padding inp"):
            # 此处还需要对inp做padding用于计算
            W = inp.size(3)
            W = int((W//self.align_num)*self.align_num+((W%self.align_num)>0)*self.align_num) 
            # employ the new W to padding inp (with zero)
            inp = torch.nn.functional.pad(inp, (0, W-inp.size(3), 0, 0), mode='constant', value=0)
        with MeasureExecutionTime(measure_name="Direct Conv2d Excution"):
            # 调用direct_conv2d.direct_conv2d函数
            output = direct_conv2d_func(inp, self.weight, self.W_bits, self.A_bits, self.MT)
        with MeasureExecutionTime(measure_name="Post process"):
            # 如果输出只要中间的一块，需要对output进行裁剪，其从0位置开始，裁剪出和inp一样大的区域
            output = self.post_func(output[:, :, 1:1+inp.size(2), 1:1+inp.size(2)])
        return output

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"W_bits={self.W_bits}, "
                f"A_bits={self.A_bits}, "
                f"MT={self.MT}, "
                f"padding={self.padding}, "
                f"stride={self.stride}, "
                f"dilation={self.dilation}, "
                f"prepare_func={self.prepare_func.__name__ if hasattr(self.prepare_func, '__name__') else self.prepare_func}, "
                f"post_func={self.post_func.__name__ if hasattr(self.post_func, '__name__') else self.post_func}, "
                f"measure_time={self.measure_time}, "
                f"keep_same_output_size={self.keep_same_output_size}, "
                f"align_w={self.align_w}, "
                f"align_num={self.align_num}, "
                f"weight_shape={tuple(self.weight.shape)})"
            )

    @staticmethod
    def from_conv2d(conv2d, W_bits, A_bits, MT=True, measure_time=False):
        # 从一个标准的Conv2d层构造一个DirectConv2d层
        return DirectConv2d(conv2d.in_channels, conv2d.out_channels, 
                            conv2d.kernel_size[0], W_bits, A_bits, MT, 
                            padding=conv2d.padding[0], stride=conv2d.stride[0], 
                            dilation=conv2d.dilation[0], measure_time=measure_time)
    
class PadConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size, 
                W_bits=3,A_bits=3,
                stride=1, 
                padding=0, dilation=1, 
                groups=1, bias=False,
                keep_same_output_size=False,
                align_w = False):
        super(PadConv2d, self).__init__()
        self.align_num = 6
        if W_bits+A_bits<3:
            self.align_num = 12
        self.conv = nn.Conv2d(in_channels, out_channels, 
                kernel_size, stride=stride, 
                padding=2, dilation=dilation, 
                groups=groups, bias=bias)
        self.align_w = align_w
        if self.align_w:
            self.alignw = AlignW(self.align_num)
        self.keep_same_output_size = keep_same_output_size
    
    def forward(self, inp):
        # with MeasureExecutionTime(measure_name="Padding inp"):
        # 此处还需要对inp做padding用于计算
        if self.align_w:
            inp = self.alignw(inp)
        # W = inp.size(3)
        # W = int((W//self.align_num)*self.align_num+((W%self.align_num)>0)*self.align_num) 
        # # employ the new W to padding inp (with zero)
        # if W!=inp.size(3):
        #     inp = torch.nn.functional.pad(inp, (0, W-inp.size(3), 0, 0), mode='constant', value=0)
        output = self.conv(inp)
        if self.keep_same_output_size:
            output = output[:, :, 1:1+inp.size(2), 1:1+inp.size(2)]
        return output
    
    @staticmethod
    def from_conv2d(conv2d, W_bits=3, A_bits=3):
        # 从一个标准的Conv2d层构造一个PadConv2d层
        return PadConv2d(conv2d.in_channels, conv2d.out_channels, 
                        conv2d.kernel_size[0], W_bits, A_bits, 
                        padding=2, stride=conv2d.stride[0], 
                        dilation=conv2d.dilation[0])

class Qint8Conv2D(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, stride=1, 
                 padding=0, dilation=1, 
                 groups=1, bias=None,
                 return_float=False,
                 engine="qnnpack"):
        super(Qint8Conv2D, self).__init__()
        torch.backends.quantized.engine = engine
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv = torch.nn.quantized.functional.conv2d
        # self.conv.weight.data = self.conv.weight.data.int()
        self.quant = lambda x: torch.quantize_per_tensor(x, 1, 0, torch.quint8)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = torch.quantize_per_tensor(weight, 1, 0, torch.qint8)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.return_float = return_float
    
    def forward(self, inp):
        # 检查，如果不是quint8的类型，则转换一下
        if not inp.dtype == torch.quint8:
            inp = self.quant(inp)
        output = self.conv(inp,self.weight,self.bias,
                           padding=self.padding, stride=self.stride,
                           groups=self.groups,dilation=self.dilation,
                           scale=1.,zero_point=0)
        if self.return_float:
            output = torch.dequantize(output)
        return output
    
    @staticmethod
    def from_float_conv2d(conv2d,copy_weight=True):
        # 从一个标准的Conv2d层构造一个Qint8Conv2D层
        qconv = Qint8Conv2D(conv2d.in_channels, conv2d.out_channels, 
                        conv2d.kernel_size[0], conv2d.stride[0], conv2d.padding[0], 
                        conv2d.dilation[0], conv2d.groups, conv2d.bias)
        if copy_weight:
            qconv.weight = torch.quantize_per_tensor(conv2d.weight.data, 1, 0, torch.qint8)
        return qconv

class PadQint8Conv2D(nn.Module):
    def __init__(self, 
                in_channels, out_channels, 
                kernel_size, 
                W_bits=3,A_bits=3,
                stride=1, 
                padding=0, dilation=1, 
                groups=1, bias=None,
                return_float=False,
                keep_same_output_size=False,
                align_w = False):
        super(PadQint8Conv2D, self).__init__()
        self.conv = Qint8Conv2D(in_channels, out_channels, kernel_size, 
                                stride, padding=2, dilation=dilation, groups=groups, 
                                bias=bias,return_float=return_float)
        self.align_num = 6
        if (W_bits+A_bits)<3:
            self.align_num = 12
        self.keep_same_output_size = keep_same_output_size
        self.align_w = align_w
        if self.align_w:
            self.alignw = AlignW(self.align_num)
    
    def forward(self, inp):
        # 此处还需要对inp做padding用于计算
        if self.align_w:
            if inp.dtype == torch.quint8:
                inp = torch.dequantize(inp)
            inp = self.alignw(inp)
        # W = inp.size(3)
        # W = int((W//self.align_num)*self.align_num+((W%self.align_num)>0)*self.align_num) 
        # # employ the new W to padding inp (with zero)
        # if W!=inp.size(3):
        #     if inp.dtype == torch.quint8:
        #         inp = torch.dequantize(inp)
        #     inp = torch.nn.functional.pad(inp, (0, W-inp.size(3), 0, 0), mode='constant', value=0)
        output = self.conv(inp)
        return output[:, :, 1:1+inp.size(2), 1:1+inp.size(2)]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"align_w={self.align_w}, keep_same_output_size={self.keep_same_output_size}, align_num={self.align_num})"

    @staticmethod
    def from_float_conv2d(conv2d,W_bits=3,A_bits=3,copy_weight=True):
        # 从一个标准的Conv2d层构造一个PadQint8Conv2D层
        qconv = PadQint8Conv2D(conv2d.in_channels, conv2d.out_channels, 
                        conv2d.kernel_size[0], W_bits, A_bits,
                        conv2d.stride[0], conv2d.padding[0], 
                        conv2d.dilation[0], conv2d.groups, conv2d.bias)
        if copy_weight:
            qconv.conv.weight = torch.quantize_per_tensor(conv2d.weight.data, 1, 0, torch.qint8)
        return qconv

# convert the float conv module to DirectConv2d
def convert_to_direct_conv2d(model, W_bits, A_bits, 
                             module_type=None,
                             module_type_name = None, 
                             prepare_func=lambda x: x.int(),
                             post_func=lambda x: x,
                             copy_weight = False,
                             verbose = True):
    
    for name, module in model.named_children():
        # 分别判断module_type 或者module_type_name来确定是否是该类型
        is_this_type = False
        if module_type is not None and isinstance(module, module_type):
            is_this_type = True
        elif module_type_name is not None and type(module).__name__ == module_type_name:
            is_this_type = True
        # print(f"module name: {name}, module type: {type(module).__name__}, is_this_type: {is_this_type}")
        if is_this_type and \
            module.kernel_size[0]==3 and module.padding[0]==1 \
                and module.stride[0]==1 and module.dilation[0]==1: # 必须要kernel size是3，padding=1, group = 1, 的才能用
            if verbose:
                print(f"convert {name} to DirectConv2d")
            direct_conv2d = DirectConv2d(module.in_channels, 
                                module.out_channels, 
                                module.kernel_size[0], 
                                W_bits, A_bits, True, 
                                module.padding[0], module.stride[0], 
                                module.dilation[0], prepare_func,post_func)
            direct_conv2d.set_quant_bits(W_bits,A_bits)
            if copy_weight:
                direct_conv2d.weight = module.weight
            setattr(model, name, direct_conv2d)
            # setattr(model, name, f"{name}_direct_conv2d")
        else:
            convert_to_direct_conv2d(module, W_bits, A_bits, 
                            module_type, module_type_name,
                            prepare_func,post_func,copy_weight,verbose)
    return model

def convert_to_pad_conv2d(model, W_bits, A_bits, 
                             module_type=None,
                             module_type_name = None, 
                             copy_weight = False,
                             verbose = True):
    
    for name, module in model.named_children():
        # 分别判断module_type 或者module_type_name来确定是否是该类型
        is_this_type = False
        if module_type is not None and isinstance(module, module_type):
            is_this_type = True
        elif module_type_name is not None and type(module).__name__ == module_type_name:
            is_this_type = True
        # print(f"module name: {name}, module type: {type(module).__name__}, is_this_type: {is_this_type}")
        if is_this_type and \
            module.kernel_size[0]==3 and module.padding[0]==1 \
                and module.stride[0]==1 and module.dilation[0]==1: # 必须要kernel size是3，padding=1, group = 1, 的才能用
            if verbose:
                print(f"convert {name} to PadConv2d")
            pad_conv2d = PadConv2d(module.in_channels, 
                                module.out_channels, 
                                module.kernel_size[0], 
                                W_bits = W_bits, A_bits = A_bits, 
                                stride= module.stride[0], 
                                dilation=module.dilation[0])
            # direct_conv2d.set_quant_bits(W_bits,A_bits)
            if copy_weight:
                pad_conv2d.weight.data.cpoy_(module.weight.data)
            setattr(model, name, pad_conv2d)
            # setattr(model, name, f"{name}_direct_conv2d")
        else:
            convert_to_pad_conv2d(module, W_bits, A_bits, 
                            module_type, module_type_name,
                            copy_weight,verbose)
    return model

class FakeFuseModuleBNReLU(nn.Module):
    def __init__(self, module):
        super(FakeFuseModuleBNReLU, self).__init__()
        self.module = module
    
    def forward(self, x):
        return self.module(x)

class FakeFuseModuleBN(nn.Module):
    def __init__(self, module):
        super(FakeFuseModuleBN, self).__init__()
        self.module = module
    
    def forward(self, x):
        return self.module(x)

class FakeFuseModuleReLU(nn.Module):
    def __init__(self, module):
        super(FakeFuseModuleReLU, self).__init__()
        self.module = module
    
    def forward(self, x):
        return self.module(x)

def fake_fuse_module_bn_relu(model,inplace = False):
    if not inplace:
        model = copy.deepcopy(model)
    pre_conv2d=False
    pre_linear=False
    pointer_module = None
    pointer_name = ""
    next = 0
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.Conv2d,DirectConv2d,PadConv2d)):
            if pre_conv2d or pre_linear:
                if next==1 and pointer_module is not None:
                    setattr(model,pointer_name,FakeFuseModuleBN(pointer_module))

            if isinstance(module, nn.Linear):
                pre_linear = True
            else:
                pre_conv2d = True
            pointer_module = module
            pointer_name = name
            next = 0
            
        elif isinstance(module, (nn.BatchNorm2d,)):
            if pre_conv2d or pre_linear:
                setattr(model, name, nn.Identity())
                next+=1
        elif isinstance(module, (nn.ReLU, )):
            if pre_conv2d or pre_linear:
                setattr(model, name, nn.Identity())
                next+=1
                if next==1 and pointer_module is not None:
                    setattr(model,pointer_name,FakeFuseModuleReLU(pointer_module))
                elif next==2 and pointer_module is not None:
                    setattr(model,pointer_name,FakeFuseModuleBNReLU(pointer_module))
                pre_linear = False
                pre_conv2d = False
                pointer_module = None
                pointer_name = ""
                next = 0
                
        else:
            if pre_conv2d or pre_linear:
                if next==1 and pointer_module is not None:
                    setattr(model,pointer_name,FakeFuseModuleBN(pointer_module))
            pre_linear = False
            pre_conv2d = False
            pointer_module = None
            pointer_name = ""
            next = 0
            fake_fuse_module_bn_relu(module,True)

    return model

# 搜索所有可以融合的模块列表，Conv|Linear+[BN]+ReLU
def find_fuse_modules(model,parent_name=""):
    pre_conv2d=False
    pre_linear=False
    pointer_module = None
    pointer_name = ""
    next = 0
    modules_to_fuse = []
    tmp_modules_to_fuse = []
    for name, module in model.named_children():
        module_name = f"{parent_name}.{name}" if parent_name!="" else name
        if isinstance(module, (nn.Linear, nn.Conv2d, DirectConv2d,PadConv2d)):
            if isinstance(module, nn.Linear):
                pre_linear = True
            else:
                pre_conv2d = True
            pointer_module = module
            pointer_name = name
            next = 0
            # module_name = f"{parent_name}.{name}" if parent_name!="" else name
            tmp_modules_to_fuse.append(module_name)
            
        elif isinstance(module, (nn.BatchNorm2d,)):
            if pre_conv2d or pre_linear:
                next+=1
                # module_name = f"{parent_name}.{name}" if parent_name!="" else name
                tmp_modules_to_fuse.append(module_name)
        elif isinstance(module, (nn.ReLU, )):
            if pre_conv2d or pre_linear:
                next+=1
                # module_name = f"{parent_name}.{name}" if parent_name!="" else name
                tmp_modules_to_fuse.append(module_name)
                modules_to_fuse.append(tmp_modules_to_fuse)
                tmp_modules_to_fuse = []
                pre_linear = False
                pre_conv2d = False
                pointer_module = None
                pointer_name = ""
                next = 0
                
        else:
            if pre_conv2d or pre_linear:
                modules_to_fuse.append(tmp_modules_to_fuse)
                # if next==1 and pointer_module is not None:
                # 	setattr(model,pointer_name,FakeFuseModuleBN(pointer_module))
            pre_linear = False
            pre_conv2d = False
            pointer_module = None
            pointer_name = ""
            next = 0
            tmp_modules_to_fuse = []
            # module_name = f"{parent_name}.{name}" if parent_name!="" else name
            modules_to_fuse += find_fuse_modules(module,module_name)
    
    # 如果他只有两个子模块的时候,我们也得处理一下
    if (pre_conv2d or pre_linear) and len(tmp_modules_to_fuse)>1:
        modules_to_fuse.append(tmp_modules_to_fuse)

    return modules_to_fuse

def fuse_module(model,inplace=False):
    model.eval()
    # 指定要融合的模块
    modules_to_fuse = find_fuse_modules(model)
    if len(modules_to_fuse)==0:
        return model
    # 融合模块
    fused_model = torch.quantization.fuse_modules(model, modules_to_fuse,inplace = inplace)
    
    return fused_model
