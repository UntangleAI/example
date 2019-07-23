import torch
import os
from torchvision import transforms
import PIL
from PIL import Image

def get_ext(filename):
    return filename.split('.')[-1].lower().strip()

def get_img_files(data_path):
    img_files = sorted(os.listdir(data_path))
    img_files = [item for item in img_files if get_ext(item) in ['jpg', 'png', 'jpeg', 'tif', 'svg', 'bit', 'bmp']]
    img_files = [os.path.join(data_path, item) for item in img_files]
    img_files = [f for f in img_files if os.path.isfile(f)]
    return(img_files)

def load_inputs(data_path, img_size=(3,224,224), is_resnet=False, transform=None):
    if(isinstance(data_path, list)):
        img_files = []
        for path in data_path:
            img_files += get_img_files(path)
    else:
        img_files = get_img_files(data_path)

    if(transform is None):
        if(is_resnet):
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    for img_path in img_files: # load images, one at a time
        try:
            input_tensor = Image.open(img_path)
            input_tensor = transform(input_tensor)
            assert input_tensor.shape == img_size
        except Exception as e:
            print("Corrupted image ignored as: {}".format(e))
            input_tensor = None

        if(input_tensor is not None):
            if (torch.cuda.is_available()):
                input_tensor = input_tensor.cuda()
        yield(img_path, input_tensor.unsqueeze(0))

def get_input_gen(root_path, subdirs=None, img_size=(3,224,224), transform=None):
    if(not subdirs):
        return(load_inputs(root_path, img_size=img_size, transform=transform))
    else:
        assert isinstance(subdirs, list), "expected list of subfolder names"
        paths = [root_path]
        for folder in subdirs:
            paths.append(os.path.join(root_path, folder))
        return(load_inputs(paths, img_size=img_size, transform=transform))
