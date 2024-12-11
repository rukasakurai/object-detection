import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def train():
    # Load data (assume preprocessed data is stored in "data_dir")
    data_dir = os.environ["DATA_DIR"]  # Set in Azure ML
    train_dataset = ...  # Load training data
    val_dataset = ...    # Load validation data

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

    # Define the model
    num_classes = 2  # Example: 1 class + background
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

    # Define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    for epoch in range(10):  # Example: 10 epochs
        model.train()
        for images, targets in train_loader:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    # Save the model
    output_dir = os.environ["OUTPUT_DIR"]  # Set in Azure ML
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "custom_model.pth"))

if __name__ == "__main__":
    train()
