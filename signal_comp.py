""" Driver script for signal estimation services.
        [1] signal estimation
        [2] signal attribution of a point
"""
__author__ = "Rahul Soni"
__copyright__ = "Untangle License"
__version__ = "1.0.6"

import os
import torch
import random
torch.set_printoptions(precision=8)
from untangle import UntangleAI
from torchvision import models
untangle_ai = UntangleAI()
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True

class MnistArgs:
    mname = 'lenet'
    batch_size = 16
    num_classes = 10
    img_size = (1,28,28)
    input_tensor = torch.randn(1,1,28,28) # provide your own input tensor
    input_tensor_true = torch.randn(28,28,1) # provide your own true input tensor / ndarray / PIL Image Obj
    data_class = None # or `None` to estimate all classes
    mode = 'attribute' # one of `estimate`, `attribute`
    topk = 1
    cmap = 'seismic'
    json = False
    hm_diff = 'joint'

class Args:
    mname = 'resnet50'
    batch_size = 16
    num_classes = 1000
    img_size = (3,224,224)
    input_tensor = torch.randn(1,3,224,224) # provide your own input tensor
    input_tensor_true = torch.randn(224,224,3) # provide your own true input tensor / ndarray / PIL Image Obj
    data_class = '945' # or `None` to estimate all classes
    mode = 'estimate' # one of `estimate`, `attribute`
    topk = 5
    cmap = 'seismic'
    json = False
    hm_diff = 'joint'

class LeNet(torch.nn.Module):
    # TODO: This isn't really a LeNet, but we implement this to be
    #  consistent with the Evidential Deep Learning paper
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = None

    def build(self):
        lenet_conv = []
        lenet_conv += [torch.nn.Conv2d(1,20, kernel_size=(5,5))]
        lenet_conv += [torch.nn.ReLU(inplace=True)]
        lenet_conv += [torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)]
        lenet_conv += [torch.nn.Conv2d(20, 50, kernel_size=(5,5))]
        lenet_conv += [torch.nn.ReLU(inplace=True)]
        lenet_conv += [torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)]

        lenet_dense = []
        lenet_dense += [torch.nn.Linear(4*4*50, 500)]
        lenet_dense += [torch.nn.ReLU(inplace=True)]
        lenet_dense += [torch.nn.Linear(500, 10)]

        self.features = torch.nn.Sequential(*lenet_conv)
        self.classifier = torch.nn.Sequential(*lenet_dense)

    def forward(self, input):
        output = self.features(input)
        output = output.view(input.shape[0], -1)
        # output = output.view(-1, 4*4*50)
        output = self.classifier(output)
        return(output)

if __name__ == '__main__':
    # args = Args()
    args = MnistArgs()
    keys = [str(item) for item in range(args.num_classes)]
    ID2Name_Map = dict(zip(keys, keys)) # backward compatibility to existing codebase

    #################################################################
    # Step-1: data data and model paths                             #
    #################################################################
    module_path = os.path.dirname(os.path.realpath(__file__))
    proj_path = os.path.abspath(os.path.join(module_path, os.pardir))
    model_signal_data_path = os.path.join(module_path, 'model_signal_data/')
    results_path = os.path.join(module_path, 'results')
    if(not os.path.exists(model_signal_data_path)):
        os.makedirs(model_signal_data_path)
    if(not os.path.exists(results_path)):
        os.makedirs(results_path)
    signal_store_path = os.path.join(model_signal_data_path, '{}_signals'.format(args.mname))

    #################################################################
    # Step-2: load model from checkpoint                            #
    #################################################################
    model_ckpt_path = os.path.join(module_path, 'lenet_mnist_model.h5')
    if(not os.path.isfile(model_ckpt_path)):
        raise FileNotFoundError(model_ckpt_path)
    print("Loading model checkpoint...", end="", flush=True)
    model = LeNet()
    model.build()
    if (torch.cuda.is_available()):
        ckpt = torch.load(model_ckpt_path)
        model.load_state_dict(ckpt)
        model = model.cuda()
    else:
        ckpt = torch.load(model_ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt)

    print("Done!")
    model.eval()

    # print("Loading model checkpoint...", end="", flush=True)
    # model = models.resnet50(pretrained=True)
    # if(torch.cuda.is_available()):
    #     model = model.cuda()
    # model.eval()
    # print("Done!")

    #################################################################
    # Step-3: If mode=`estimate`, estimate the activation stats     #
    #################################################################
    if(args.mode == 'estimate'):
        def train_loader_fun(class_i):
            loader, _ = untangle_ai.load_mnist_per_class(batch_size=args.batch_size, data_class=class_i)

            # data_path = '../imagenet_test/data/train/'
            # SYNSET_MAP = {'945': 'n07720875'} # bell pepper example
            # loader, _ = untangle_ai.load_imagenet_per_class(batch_size=args.batch_size,
            #                                 data_path=data_path,
            #                                 synset_map=SYNSET_MAP, data_class=class_i,
            #                                 is_resnet='True')
            return(loader)


        untangle_ai.estimate_signals(model, signal_store_path, train_loader_fun, args)

    #################################################################
    # Step-4: If mode=`eval_point`, evaluate uncertainty of a point #
    #################################################################
    elif(args.mode == 'attribute'):
        data_gen = untangle_ai.mnist_data_gen(args.batch_size, results_path, gen_images=False)
        for img_path, input_tensor in tqdm(data_gen):
            for idx in range(len(img_path)):
                input_tensor_single = input_tensor[idx][None, :, :, :]
                input_tensor_true = input_tensor[idx].permute(1, 2, 0) # expected shape (H,W,C)
                out_prefix = os.path.join(results_path, '{}_signals'.format(img_path[idx]))
                untangle_ai.attribute_signals(model, input_tensor_single, input_tensor_true, signal_store_path,
                          ID2Name_Map, args, out_prefix)

    else:
        raise ValueError('invalid mode argument `{}`.'
                         'Must be one of `estimate`, `attribute`'.format(args.mode))

    print("Execution complete")
