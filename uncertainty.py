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
from tqdm import tqdm
from copy import deepcopy
torch.set_printoptions(precision=8)
from time import time
from datetime import datetime
from untangle import UntangleAI
from torchvision import models

untangle_ai = UntangleAI()

class MnistArgs:
    mname = 'lenet'
    batch_size = 16
    num_classes = 10
    img_size = (1,28,28)
    input_tensor = torch.randn(1,1,28,28) # provide your own input tensor
    data_class = None # or `None` to model uncertainty for all classes
    mode = 'model' # one of `model`, `eval_point`, `rank`
    out_thres = 0.3 # outlier threshold (default: `30%% of cluster2cluster distance`)
    dir = '../Kannada/Img/GoodImg/Bmp/Sample004'
    rank_type = 'global' # in case of class by clas ranking, pass `local` rank_type
    metric = 'prob' # for future additions. Current default set to `prob`

class Args:
    mname = 'vgg16'
    batch_size = 16
    num_classes = 1000
    img_size = (3,224,224)
    torch.manual_seed(0)
    input_tensor = torch.randn(1,3,224,224) # provide your own input tensor
    data_class = '623' # or `None` to model uncertainty for all classes
    mode = 'eval_point' # one of `model`, `eval_point`, `rank`
    out_thres = 0.3 # outlier threshold (default: `30%% of cluster2cluster distance`)
    dir = '../Kannada/Img/GoodImg/Bmp/Sample004'
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

def model_uncertainty(base_model, uncertainty_store_path, train_loader_fun, args):
    ############################################################
    # Train accordingly to the following rule                  #
    #   [1] If data-class is specified, train that class       #
    #   [2] Else train all classes                             #
    ############################################################
    stats_store_path = "temp_stats_store_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".h5"

    if(args.data_class is not None):
        class_list = [args.data_class]
    else:
        class_list = list(range(args.num_classes))
        class_list = [str(item) for item in class_list]

    for class_i in class_list:
        print("Estimating uncertainty for class: `{}`".format(class_i))
        model = deepcopy(base_model)

        uncrt_modeller = untangle_ai.uncertainty_estimator(model, args.img_size, stats_store_path)
        train_loader = train_loader_fun(class_i)
        for (input_tensor, target) in tqdm(train_loader):
            if(input_tensor is None):
                continue
            else:
                if(torch.cuda.is_available()):
                    input_tensor = input_tensor.cuda()
                uncrt_modeller.update_stats(input_tensor)
        uncrt_modeller.close_stats_files_if_open()

        #############################################################
        # Save pattern for future investigation:                    #
        #############################################################
        uncrt_modeller.compute_stats()
        uncrt_path_for_class_i = uncertainty_store_path + '_class_{}.pkl'.format(class_i)
        uncrt_modeller.save_activations(uncrt_path_for_class_i)
        print('uncertainty stats saved in path: {}'.format(uncrt_path_for_class_i))
        del uncrt_modeller
        del model

    # Clean-up of temp file
    if(os.path.exists(stats_store_path)):
        os.remove(stats_store_path)

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
    model_uncrt_data_path = os.path.join(module_path, 'model_uncrt_data/')
    results_path = os.path.join(module_path, 'results')
    if(not os.path.exists(model_uncrt_data_path)):
        os.makedirs(model_uncrt_data_path)
    if(not os.path.exists(results_path)):
        os.makedirs(results_path)
    uncertainty_store_path = os.path.join(model_uncrt_data_path, '{}_uncertainty'.format(args.mname))

    #################################################################
    # Step-2: load model from checkpoint                            #
    #################################################################
    #model_ckpt_path = os.path.join(module_path, 'lenet_mnist_model.h5')
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
    # model = models.vgg16(pretrained=True)
    # if(torch.cuda.is_available()):
    #     model = model.cuda()
    # model.eval()
    # print("Done!")

    #################################################################
    # Step-3: If mode=`model`, model uncertainty for specified class#
    #################################################################
    if(args.mode == 'model'):
        def train_loader_fun(class_i):
            loader, _ = untangle_ai.load_mnist_per_class(batch_size=args.batch_size, data_class=class_i)

            # data_path = '../imagenet_test/data/train/'
            # SYNSET_MAP = {'623': 'n03658185'}
            # loader, _ = data_util.load_imagenet_per_class(batch_size=args.batch_size,
            #                                 data_path=data_path,
            #                                 synset_map=SYNSET_MAP, data_class=class_i,
            #                                 is_resnet='False')
            return(loader)

        model_uncertainty(model, uncertainty_store_path, train_loader_fun, args)

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
        data_name = args.dir.split('/')[-1].strip()
        base_out_path = os.path.join(results_path, data_name)
        if(not os.path.exists(base_out_path)):
            os.makedirs(base_out_path)

        t1 = time()
        untangle_ai.rank_data(model, uncertainty_store_path, base_out_path, ID2Name_Map, args)
        print("completed in: {}".format(time() - t1))

    else:
        raise ValueError('invalid mode argument `{}`.'
                         'Must be one of `model`, `eval_point`, `rank`'.format(args.mode))

    print("Execution complete")
