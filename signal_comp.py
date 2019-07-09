""" Driver script for signal estimation services.
        [1] signal estimation
        [2] signal attribution of a point
"""
__version__ = "0.1.0"
__status__ = "Development"
__date__ = "28/June/2019"

import os
import torch
from tqdm import tqdm
from copy import deepcopy
torch.set_printoptions(precision=8)
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
    input_tensor_true = torch.randn(28,28,1) # provide your own true input tensor / ndarray / PIL Image Obj
    data_class = None # or `None` to estimate all classes
    mode = 'attribute' # one of `estimate`, `attribute`
    topk = 5
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
    mode = 'attribute' # one of `estimate`, `attribute`
    topk = 1
    cmap = 'seismic'
    json = False
    hm_diff = 'joint'

def estimate_signals(base_model, signal_store_path, train_loader_fun, args):
    '''
    estimates signals class by class
    :param base_model: pytorch model object
    :param signal_store_path: path to store signal for each class
    :param train_loader_fun: data loader function that takes class_idx as arg
    :param args: data specific params (ref to MnistArgs class)
    :return: None
    '''

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
        print("Estimating signals for class: `{}`".format(class_i))
        model = deepcopy(base_model)

        signal_estimator = untangle_ai.signal_estimator(model=model, img_size=args.img_size,
                                             stats_store_path=stats_store_path)

        train_loader = train_loader_fun(class_i)
        for (input_tensor, target) in tqdm(train_loader):
            if(input_tensor is None):
                continue
            else:
                if(torch.cuda.is_available()):
                    input_tensor = input_tensor.cuda()
                signal_estimator.update_stats(input_tensor)
        signal_estimator.close_stats_files_if_open()

        #############################################################
        # Save signals for future investigation:                    #
        #############################################################
        signal_estimator.compute_signals()
        signal_path_for_class_i = signal_store_path + '_class_{}.pkl'.format(class_i)
        signal_estimator.save_signals(signal_path_for_class_i)
        print('signals saved in path: {}'.format(signal_path_for_class_i))
        del signal_estimator
        del model

    # Clean-up of temp file
    if(os.path.exists(stats_store_path)):
        os.remove(stats_store_path)

def _attribute(base_model, base_input_tensor, signals_path, class_idx, args):

    model = deepcopy(base_model)
    input_tensor = deepcopy(base_input_tensor)
    input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
    model.eval()
    outp = model(input_tensor)
    one_hot_output = torch.zeros_like(outp).to(outp.device)
    one_hot_output[:, class_idx] = 1.0
    model.zero_grad()

    signal_store_path = signals_path + '_class_{}.pkl'.format(class_idx)
    if (not os.path.isfile(signal_store_path)):
        raise FileNotFoundError(signal_store_path)
    print("Using signals: {}".format(signal_store_path))

    signalModel = untangle_ai.signal_attribution(model, args.img_size, signal_store_path)
    signalModel.update_model()

    outp.backward(gradient=one_hot_output)
    signal = input_tensor.grad
    return(signal.detach())

def attribute_signals(base_model, base_input_tensor, input_tensor_true,
                  signals_path, ID2Name_Map, args, out_name='signal.JPEG'):
    '''
    get signal attributes for an input image
    :param base_model: pytorch model object
    :param base_input_tensor: torch tensor of shape: (1,C,H,W)
    :param input_tensor_true: torch tensor / ndaray / PIL Image obj, of shape (H,W,C)
    :param signals_path: path where signal estimates reside
    :param args: args of type Args
    :param out_name: optional, out_name of visualizations
    :return: None
    '''

    if (torch.cuda.is_available()):
        base_input_tensor = base_input_tensor.cuda()

    topk_logits, topk_prob, topk_classes, topk_indices = untangle_ai.get_topk_pred(base_model,
                                                    base_input_tensor, ID2Name_Map, args.topk)
    print("Top-{} class(es): {}".format(args.topk, topk_classes))
    print("Top-{} indices(s): {}".format(args.topk,topk_indices))
    print("Top-{} logit(s): {}".format(args.topk,topk_logits))
    print("Top-{} probs(s): {}".format(args.topk,topk_prob))

    topk_gradients = []
    for idx, class_idx in enumerate(topk_indices):
        print("-" * 70)
        print("{} Signal visualization for class: {} [{}]".
              format(" " * 4, topk_classes[idx], topk_indices[idx]))
        print("-" * 70)
        gradient = _attribute(base_model, base_input_tensor, signals_path, class_idx, args)
        topk_gradients.append(gradient)

    # signal heatmaps
    rgb_heatmaps, channel_heatmaps = untangle_ai.visualization().get_joint_heatmaps(topk_gradients, args)
    out_path = os.path.join(module_path, out_name)
    img_size = (args.img_size[1], args.img_size[2], args.img_size[0])  # channel last
    if(args.json):
        untangle_ai.visualization().save_joint_heatmaps(rgb_heatmaps, channel_heatmaps, topk_prob,
                                           topk_classes, out_path, img_size)
    else:
        untangle_ai.visualization().visualize_joint_heatmaps(input_tensor_true, rgb_heatmaps, channel_heatmaps,
                                                out_path, img_size)

    # difference heatmaps
    rgb_heatmaps, channel_heatmaps = untangle_ai.visualization().get_joint_diff_heatmaps(topk_gradients, args)
    out_path = os.path.join(module_path, out_name)
    if(args.json):
        untangle_ai.visualization().save_joint_diff_heatmaps(rgb_heatmaps, channel_heatmaps, topk_prob,
                                           topk_classes, out_path, img_size)
    else:
        untangle_ai.visualization().visualize_joint_diff_heatmaps(input_tensor_true, rgb_heatmaps, channel_heatmaps,
                                                     out_path, img_size)

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

        estimate_signals(model, signal_store_path, train_loader_fun, args)

    #################################################################
    # Step-4: If mode=`eval_point`, evaluate uncertainty of a point #
    #################################################################
    elif(args.mode == 'attribute'):
        input_tensor = args.input_tensor # expected shape (1,C,H,W)
        input_tensor_true = args.input_tensor_true # expected shape (H,W,C)
        out_name = 'results/{}_{}'.format(args.mname, 'test_img')

        attribute_signals(model, input_tensor, input_tensor_true, signal_store_path,
                          ID2Name_Map, args, out_name)

    else:
        raise ValueError('invalid mode argument `{}`.'
                         'Must be one of `estimate`, `attribute`'.format(args.mode))

    print("Execution complete")
