""" Driver script for uncertainty modelling services.
        [1] uncertainty modelling
        [2] uncertainty evaluation of a point
        [3] uncertainty ranking of a dataset
"""
__copyright__ = "Untangle License"
__version__ = "1.0.1"
__status__ = "Development"
__date__ = "28/June/2019"

import os
import argparse
import torch
import random
from tqdm import tqdm
from copy import deepcopy
torch.set_printoptions(precision=8)
from time import time
from datetime import datetime
from untangle import UntangleAI
from torchvision import models
untangle_ai = UntangleAI()

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True

class MnistArgs:
    mname = 'lenet'
    batch_size = 64
    num_classes = 10
    img_size = (1,28,28)
    input_tensor = torch.randn(1,1,28,28) # provide your own input tensor
    data_class = None # or `None` to model uncertainty for all classes
    mode = 'rank' # one of `model`, `eval_point`, `rank`
    outlier_threshold = -3 # outlier threshold (default: `30%% of cluster2cluster distance`)
    uncertainty_threshold = 0.2
    dir = '../Kannada/Img/GoodImg/Bmp/Sample004'
    rank_type = 'global' # in case of class by clas ranking, pass `local` rank_type
    metric = 'prob' # 'prob' or 'similarity'
    sigmoid_node = False

class Args:
    mname = 'vgg16'
    batch_size = 16
    num_classes = 1000
    img_size = (3,224,224)
    torch.manual_seed(0)
    input_tensor = torch.randn(1,3,224,224) # provide your own input tensor
    data_class = '0' # or `None` to model uncertainty for all classes
    mode = 'model' # one of `model`, `eval_point`, `rank`
    out_thres = 0.3 # outlier threshold (default: `30%% of cluster2cluster distance`)
    dir = '../imagenet_test/data/ranking_test_data/'
    rank_type = 'global' # in case of class by clas ranking, pass `local` rank_type
    metric = 'prob' # for future additions. Current default set to `prob`

class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

class LeNet(torch.nn.Module):
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
    #args = Args()
    args = MnistArgs()
    keys = [str(item) for item in range(args.num_classes)]
    ID2Name_Map = dict(zip(keys, keys)) # backward compatibility to existing codebase

    #################################################################
    # Step-1: data data and model paths                             #
    #################################################################
    module_path = os.path.dirname(os.path.realpath(__file__))
    proj_path = os.path.abspath(os.path.join(module_path, os.pardir))
    model_uncrt_data_path = os.path.join(module_path, 'model_uncrt_data/')
    results_path = os.path.join(module_path, 'results')
    if(not os.path.exists(model_uncrt_data_path)):
        os.makedirs(model_uncrt_data_path)
    if(not os.path.exists(results_path)):
        os.makedirs(results_path)
    uncertainty_store_path = os.path.join(model_uncrt_data_path, '{}_uncertainty'.format(args.mname))

    ################################################################
    # Step-2: load model from checkpoint                           #
    ################################################################
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

    #print("Loading model checkpoint...", end="", flush=True)
    #model = models.vgg16(pretrained=True)
    #if(torch.cuda.is_available()):
    #    model = model.cuda()
    #model.eval()
    #print("Done!")

    #################################################################
    # Step-3: If mode=`model`, model uncertainty for specified class#
    #################################################################
    if(args.mode == 'model'):
        def train_loader_fun(class_i):
            loader, _ = untangle_ai.load_mnist_per_class(batch_size=args.batch_size, data_class=class_i)

            #data_path = '../imagenet_test/data/train/'
            #SYNSET_MAP = {'0': 'n01440764'}
            #loader, _ = untangle_ai.load_imagenet_per_class(batch_size=args.batch_size,
            #                                data_path=data_path,
            #                                synset_map=SYNSET_MAP, data_class=class_i,
            #                                is_resnet='False')
            return(loader)

        untangle_ai.model_uncertainty(model, uncertainty_store_path, train_loader_fun, args)

    #################################################################
    # Step-4: If mode=`eval_point`, evaluate uncertainty of a point #
    #################################################################
    elif(args.mode == 'eval_point'):
        # expect to be of size (1,C,H,W)
        input_tensor = args.input_tensor
        assert len(input_tensor.shape) == 4, "input tensor must be of shape: (1,C,H,W)"
        if (torch.cuda.is_available()):
            input_tensor = input_tensor.cuda()
        untangle_ai.eval_point(model, input_tensor, uncertainty_store_path, ID2Name_Map, args)

    #################################################################
    # Step-5: If mode=`rank`, rank a dataset on uncertainty         #
    #################################################################
    elif(args.mode == 'rank'):

        #data_gen = untangle_ai.load_batched_inputs(args.dir, batch_size=50, is_resnet=False)
        #
        #label_path = '../imagenet_test/data/label_mapping.txt'
        #ID2Name_Map = {}
        #with open(label_path) as f:
        #    for line in f:
        #        (idx, name) = line.split(':')
        #        ID2Name_Map[str(idx)] = name.strip()
        data_gen = untangle_ai.mnist_data_gen(args.batch_size, results_path, gen_images=False)

        t1 = time()
        untangle_ai.rank_data(model, data_gen, uncertainty_store_path, results_path, ID2Name_Map, args)
        print("completed in: {}".format(time() - t1))

    print("Execution complete")
