""" Driver script for concept extraction services."""

__version__ = "0.1.0"
__status__ = "Development"
__date__ = "08/July/2019"

import torchvision.models as models
from untangle import UntangleAI
untangle_ai = UntangleAI()

class Args:
    mname = 'vgg16'
    act_type = 'flatten'
    train_data_path = 'floral_patterns_dataset/data/' # provide path containing concept images
    test_data_path = 'randomised_test_dataset/' # probide path containing test images
    save_path = 'ranked_images_in_randomised_data' # created ranked csv file with this name
    batch_size = 16
    img_size = (3,224,224)

def get_concepts(model, args):
    # train_loader / test_loader: returns (img_paths, input_tensor)
    # alternatively, provide your own data loaders
    train_loader = untangle_ai.load_batched_inputs(args.train_data_path, batch_size=args.batch_size)
    test_loader = untangle_ai.load_batched_inputs(args.test_data_path, batch_size=50)

    # create an instance of concept extractor
    concept_extractor = untangle_ai.concept_extractor(model, img_size=args.img_size)

    # extract concepts from the model
    concept_extractor.learn_linear_concepts(train_loader)

    for imgs, input_tensors in test_loader:
        concept_extractor.rank_images_per_batch(input_tensors, imgs)

    concept_extractor.save_ranking(args.save_path)

if __name__ == '__main__':
    args = Args()
    model = models.vgg16(pretrained=True)
    get_concepts(model, args)
