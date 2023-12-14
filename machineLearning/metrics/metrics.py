# code for metrics used to evaluate models
import torch
import matplotlib.pyplot as plt

def accuracy(y_pred, y_true):
    """
    Calculates the accuracy of the predictions
    """
    return torch.mean(((y_pred > 0.5) == y_true.byte()).float())

def plot_continuous_performance(epoch, true, pred):
    # scatter plot of true vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(true, pred, color='blue', alpha=0.5)
    plt.title('True vs. Predicted Values in Regression')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    # set x axis range
    plt.xlim(0, 1)
    # set y axis range
    plt.ylim(0, 1)
    # draw a diagonal line
    plt.plot([-100, 100], [-100, 100], color='red', linestyle='--')
    plt.grid(True)
    plt.savefig(f"{epoch}_true_vs_pred.png")
    return plt