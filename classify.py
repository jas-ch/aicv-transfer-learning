import collections
import torch, torchvision
import os, sys, time
import glob
import argparse
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(37)

weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
conv = torchvision.models.convnext_base(weights = weights)

inference_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.CenterCrop(224), 
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

conv.classifier = torch.nn.Identity()
model = torch.nn.Sequential(collections.OrderedDict([
    ("convnext_base", conv),
    ("flatten", torch.nn.Flatten()),
    ("final", torch.nn.Linear(1024, 29)),
#    ("softmax", torch.nn.Softmax(dim = 1)),
]))

# Model moved to GPU
model = model.to(device)

# optimizer = torch.optim.Adam(param_groups)
loss_fn = torch.nn.CrossEntropyLoss()

lr = 0.001
param_groups = [
    {'params': model.convnext_base.parameters(), 'requires_grad': False},
    {'params': model.final.parameters(), 'lr': lr},
]

def find_latest_checkpoint():
    """Find the latest epoch checkpoint from 1-5"""
    all_checkpoints = glob.glob('checkpoint_epoch*.tar')
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)

    for checkpoint in all_checkpoints:
        fh = torch.load(checkpoint, weights_only=True, map_location=device)
        if 'model_state_dict' not in fh:
            print(f"incomplete checkpoint {checkpoint}.\n")
        else:
            print(f"found latest checkpoint {checkpoint}")
            return fh, checkpoint
    return None, None

checkpoint, cp_filename = find_latest_checkpoint() 
if checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
#   optimizer.load_state_dict(checkpoint['optim_state_dict'])
    print(f"{cp_filename} model weights loaded.")

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']
def classify(model, input_dir, transforms, loss_fn, classes, device, output_path, output_file):
    model.eval()

    loss = 0.0

    images = glob.glob(os.path.join(input_dir+"/*.jpg"))
    if output_path != '':
        output_file = os.path.join(output_path+'/'+output_file)
    with open(output_file, 'w') as f:
        f.write(f"results for running model on data in {input_dir}\n")
        for image in images:
            im = Image.open(image).convert('RGB')
            input_tensor = transforms(im).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                prob = torch.softmax(outputs, dim=1)
                top_prob, top_class_idx = torch.max(prob, 1)

            label = os.path.basename(image).split('_')[0]
            predict = classes[top_class_idx.item()]
            loss = loss if (label == predict) else loss+1

            result = {
                'filename': image, 
                'predicted_class': predict,
                'confidence': top_prob.item()
            }

            f.write(json.dumps(result))
            f.write('\n')

        print(f"prediction accuracy = {1 - loss/len(images)}.")
        f.write(f"prediction accuracy = {1 - loss/len(images)}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image_path', help='path to the input images folder', default=None)
    parser.add_argument('-p', '--output_path', help='path for output', default='')
    parser.add_argument('-o', '--output_file', help='output filename', default='output.txt')
    args = parser.parse_args()

    if args.input_image_path:
        classify(model, args.input_image_path, inference_tf, loss_fn, classes, device, args.output_path, args.output_file)
    else:
        print(f"missing input image path...")

