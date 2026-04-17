import collections
import torch, torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

epochs = 5

torch.manual_seed(37)

weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
conv = torchvision.models.convnext_base(weights = weights)

def get_dataloaders(batch_size: int, train_proportion: float) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """load in train/val datasets (split full data)"""
    path_to_data = "asl/asl_alphabet_train/asl_alphabet_train"
    # path_to_test = "asl/asl_alphabet_test/asl_alphabet_test"

    # Reference: Documented from Pytorch TorchVision Models Documentation 
    #            https://pytorch.org/vision/stable/models.html
    # "All pre-trained models expect input images normalized in the same way, 
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W 
    # are expected to be at least 224. The images have to be loaded in to a range 
    # of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and 
    # std = [0.229, 0.224, 0.225]."
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.CenterCrop(224), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    full_set = torchvision.datasets.ImageFolder(
        path_to_data,
        transform = transforms,
    )
    train_set, val_set = torch.utils.data.random_split(
        full_set,
        lengths = [train_proportion, 1-train_proportion],
    )

    num_workers = 4

    train = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # test = torch.utils.data.DataLoader(test_set, batch_size = batch_size)
    return train, val

train, val = get_dataloaders(32, 0.8)

print(f"Number of training batches: {len(train)}")
print(f"Number of val batches: {len(val)}")

# conv.classifer = torch.nn.Identity()
conv.classifier = torch.nn.Identity()

############# Conv structure ###############
print(f"\nOriginal classifier: {conv.classifier}")
dummy = torch.randn(1, 3, 224, 224).to(device)
conv_test = conv.to(device)
with torch.no_grad():
    output = conv_test(dummy)
    print(f"Original ConvNeXt output shape: {output.shape}")

model = torch.nn.Sequential(collections.OrderedDict([
    ("convnext_base", conv),
    ("flatten", torch.nn.Flatten()), 
    ("final", torch.nn.Linear(1024, 29)),
#    ("softmax", torch.nn.Softmax(dim = 1)),
]))

# Model moved to GPU
model = model.to(device)

lr = 0.001         # search here (grid?), 0.1 0.01 0.001 0.0001
param_groups = [
    {'params': model.convnext_base.parameters(), 'requires_grad': False},
    {'params': model.final.parameters(), 'lr': lr},
]

optimizer = torch.optim.Adam(param_groups)
loss_fn = torch.nn.CrossEntropyLoss()

def validate(model, val_loader, loss_fn, device):
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
                for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)

                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

        model.train()
        return val_loss/len(val_loader), val_correct/val_total

for i in range(epochs):
    batch_losses = []
    batch_accuracies = []
    print(f"---- EPOCH {i+1} ----")

    model.train()
    for batch_idx, (image_batch, label_batch) in enumerate(train):
        image_batch, label_batch = image_batch.to(device, non_blocking=True), label_batch.to(device, non_blocking=True)
        preds = model(image_batch)
        loss = loss_fn(preds, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        cur_loss = sum(batch_losses)/len(batch_losses)
        this_batch_acc = int(sum(preds.argmax(1)==label_batch)) / len(label_batch)
        batch_accuracies.append(this_batch_acc)
        cur_acc = sum(batch_accuracies)/len(batch_accuracies)

        print("Train:", end="\t\t")
        print(f"Batch: {len(batch_losses)}", end="\t")
        print(f"Loss: {cur_loss:.4f}", end="\t")
        print(f"Accuracy: {cur_acc:.4f}", end="\r")

    # epoch statistics
    epoch_loss = sum(batch_losses) / len(batch_losses)
    epoch_acc = sum(batch_accuracies) / len(batch_accuracies)

    # validation phase
    val_loss, val_acc = validate(model, val, loss_fn, device)
    print(f"\nEpoch {i+1} Summary:")
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # save
    PATH = f'checkpoint_epoch{i+1}.tar'
    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'train_acc': epoch_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, PATH)

    print(f"Checkpoint saved to {PATH}\n")

print("Training complete!")
print(f"Final model is on device: {next(model.parameters()).device}")