# main code for training and evaluating models
import os
import mlflow
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from dataloading.loaddata import prepare_data, CustomDataset, timeseries_split, random_split, timeseries_eval_split
from metrics.metrics import accuracy, plot_continuous_performance
from models.cnn import CNN
from models.resnet import ResNet
from models.vit import ViT
from featureengineering.transformations import CNNPreprocess, CNNPreprocessGrey, ResNetPreprocess, ResNetPreprocessGrey, ViTPreprocess, ViTPreprocessGrey




def train_model(model, type, train_loader, val_loader, criterion, optimizer, num_epochs):
    try:
        mlflow.set_experiment(f"Final-{type}")
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("criteria", args.criteria)
        mlflow.log_param("optimizer", args.optimizer)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("model", args.model)
        mlflow.log_param("filter", args.filter)
        mlflow.log_param("training_split", args.training_split)
        mlflow.log_param("exp_type", args.training_split)
        epoch_cntr = 1
        for epoch in range(num_epochs):
            avg_loss = []
            if type == "binary": avg_acc = []
            for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training  ", ncols=100):
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # resnet model Forward passes and calculate loss
                predictions = model(inputs)
                predictions = predictions.squeeze()

                loss = criterion(predictions, targets)
                avg_loss.append(loss.item())
                
                if type == "binary": 
                    acc = accuracy(predictions, targets)
                    avg_acc.append(acc.item())

                loss.backward()
                optimizer.step()

            # Validation
            avg_loss_val = []
            if type == "binary": avg_acc_val = []
            if type == "continuous": true, pred = [], []
            for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", ncols=100):
                predictions = model(inputs)
                predictions = predictions.squeeze()
                loss = criterion(predictions, targets)
                avg_loss_val.append(loss.item())

                if type == "binary": 
                    acc = accuracy(predictions, targets)
                    avg_acc_val.append(acc.item())
                elif type == "continuous":
                    # targets to array
                    true.extend(targets.cpu().detach().numpy())
                    pred.extend(predictions.cpu().detach().numpy())

            if type == "continuous":
                plot_continuous_performance(epoch, true, pred)
                mlflow.log_artifact(f"{epoch}_true_vs_pred.png")

            print(f"Epoch: {epoch+1}/{num_epochs}, loss: {sum(avg_loss)/len(avg_loss)}, val_loss: {sum(avg_loss_val)/len(avg_loss_val)}")
            
            mlflow.log_metric("loss", sum(avg_loss)/len(avg_loss), step=epoch)
            mlflow.log_metric("val_loss", sum(avg_loss_val)/len(avg_loss_val), step=epoch)

            if type == "binary": 
                mlflow.log_metric("accuracy", sum(avg_acc)/len(avg_acc), step=epoch)
                mlflow.log_metric("val_accuracy", sum(avg_acc_val)/len(avg_acc_val), step=epoch)

            epoch_cntr += 1

        mlflow.log_param("num_epochs", epoch_cntr)
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
    except Exception as e:
        print(e)
        mlflow.log_param("num_epochs", epoch_cntr)
        torch.cuda.empty_cache()
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet", help="Model to use")
    parser.add_argument("--type", type=str, default="binary", help="Model type, either binary or continuous")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--criteria", type=str, default="l1", help="Loss function")
    parser.add_argument("--optimizer", type=str, default="sgd", help="Optimizer, either adam or sgd")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--csv_dir", type=str, help="CSV file directory")
    parser.add_argument("--eval_csv_dir", type=str, help="CSV file directory for evaluation")
    parser.add_argument("--img_dir", type=str, help="Image folder directory")
    parser.add_argument("--eval_img_dir", type=str, help="Image folder directory for evaluation")
    parser.add_argument("--filter", type=str, default="None", help="Filter to use, options are None, grey, hess")
    parser.add_argument("--training_split", type=str, default="timeseries", help="Training split, options are timeseries or evaluation (uses timeseries and test set for evaluation))")

    args = parser.parse_args()

    # Load data
    binary_dataset, continuous_dataset = prepare_data(args.csv_dir, args.img_dir, args.filter)
    if args.training_split == "evaluation": 
        binary_dataset_eval, continuous_dataset_eval = prepare_data(args.eval_csv_dir, args.eval_img_dir, args.filter)

    # preprocessing
    if args.model == "cnn":
        if args.filter == "None": preprocess = CNNPreprocess()
        elif args.filter == "grey": preprocess = CNNPreprocessGrey()
        elif args.filter == "hess": preprocess = CNNPreprocess()
    elif args.model == "resnet":
        if args.filter == "None": preprocess = ResNetPreprocess()
        elif args.filter == "grey": preprocess = ResNetPreprocessGrey()
        elif args.filter == "hess": preprocess = ResNetPreprocess()
    elif args.model == "vit":
        if args.filter == "None": preprocess = ViTPreprocess()
        elif args.filter == "grey": preprocess = ViTPreprocessGrey()
        elif args.filter == "hess": preprocess = ViTPreprocess()

    # Create datasets
    if args.type == "binary":
        if args.training_split == "evaluation":
            train = timeseries_eval_split(binary_dataset)
            val = timeseries_eval_split(binary_dataset_eval)
        elif args.training_split == "timeseries":
            train, val = timeseries_split(binary_dataset)
        else:
           train, val = random_split(binary_dataset)
    elif args.type == "continuous":
        if args.training_split == "evaluation":
            train = timeseries_eval_split(continuous_dataset)
            val = timeseries_eval_split(continuous_dataset_eval)
        elif args.training_split == "timeseries":
            train, val = timeseries_split(continuous_dataset)
        else:
           train, val = random_split(continuous_dataset)

    train_dataset = CustomDataset(train, transform=preprocess)
    val_dataset = CustomDataset(val, transform=preprocess)

    # Create dataloaders
    if args.training_split == "random":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # select input channels
    if args.filter == "grey" or args.filter == "hess":
        num_channels = 1
    else:
        num_channels = 3

    # Create model
    if args.model == "cnn":
        model = CNN(num_channels=num_channels)
    elif args.model == "resnet":
        model = ResNet(num_channels=num_channels)
    elif args.model == "vit":
        model = ViT(num_channels=num_channels)

    # Loss functions
    if args.criteria == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.criteria == "mse":
        criterion = nn.MSELoss()
    elif args.criteria ==  "l1":
        criterion = nn.L1Loss()
        
    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # Train model
    train_model(model, args.type, train_loader, val_loader, criterion, optimizer, args.num_epochs)
