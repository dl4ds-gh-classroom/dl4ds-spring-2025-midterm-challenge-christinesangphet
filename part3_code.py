import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision import models
import wandb
from tqdm import tqdm


################################################################################
# Model Definition - ResNet-50
################################################################################
class ResNet50Model(nn.Module):
    def __init__(self, num_classes=100, dropout_prob=0.3): # Suggested by ChatGPT to experiment dropout
        super(ResNet50Model, self).__init__()
        # Load pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the fully connected layer to match the number of classes in CIFAR-100
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


################################################################################
# Training Function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Validation Function
################################################################################
def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    ############################################################################
    # Configuration Dictionary
    ############################################################################
    CONFIG = {
        "model": "ResNet50", 
        "batch_size": 512, # Suggested by ChatGPT to experiment 
        "learning_rate": 0.001,
        "epochs": 30, # Suggested by ChatGPT to experiment 
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 26,
    }

    torch.manual_seed(CONFIG["seed"])
    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    # Data Transformation
    ############################################################################
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    ############################################################################
    # Data Loading
    ############################################################################
    trainset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    testset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=CONFIG["batch_size"],
        shuffle=True, 
        num_workers=CONFIG["num_workers"]
    )
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=CONFIG["batch_size"],
        shuffle=False, 
        num_workers=CONFIG["num_workers"]
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=CONFIG["batch_size"],
        shuffle=False, 
        num_workers=CONFIG["num_workers"]
    )
    
    ############################################################################
    # Instantiate model
    ############################################################################
    model = ResNet50Model(num_classes=100) 
    model = model.to(CONFIG["device"])

    print("\nModel summary:")
    print(f"{model}\n")

    ############################################################################
    # Loss Function and Optimizer
    ############################################################################
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Suggested by ChatGPT to experiment 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Suggested by ChatGPT to experiment weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize wandb
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    ############################################################################
    # Training Loop
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    ############################################################################
    # Evaluation
    ############################################################################
    import eval_cifar100
    import eval_ood

    # Evaluation on Clean CIFAR-100 Test Set
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # Evaluation on OOD
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # Create Submission File (OOD)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission3_ood.csv", index=False)
    print("submission3_ood.csv created successfully.")

if __name__ == '__main__':
    main()