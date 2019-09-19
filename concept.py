""" Driver script for concept extraction services."""
__copyright__ = "Untangle License"
__version__ = "1.0.1"
__status__ = "Development"

import torchvision.models as models
from untangle import UntangleAI
untangle_ai = UntangleAI()

class Args:
    mname = 'vgg16'
    train_data_path = './floral_patterns_dataset/' # provide path containing concept images
    test_data_path = './randomised_test_dataset' # provide path containing test images
    save_path = './results/concept_results/' # provide path to save ranked concepts list
    batch_size = 16
    img_size = (3,224,224)

if __name__ == '__main__':
    args = Args()
    model = models.vgg16(pretrained=True)

    # train_loader / test_loader: returns (img_paths, input_tensor)
    # alternatively, provide your own data loaders
    train_loader = untangle_ai.load_batched_inputs(args.train_data_path, batch_size=args.batch_size)
    test_loader = untangle_ai.load_batched_inputs(args.test_data_path, batch_size=50)
    untangle_ai.extract_concept_images(model, train_loader, test_loader, args)
