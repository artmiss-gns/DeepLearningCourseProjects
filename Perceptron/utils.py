import matplotlib.pyplot as plt
import numpy as np
import torch

def plotter(x, y_true, y_pred=None, **kwargs) :
    plt.figure(figsize=(5, 4), layout="constrained")
    plt.scatter(x, y_true, c="g", s=4, label="y_true", **kwargs)
    if y_pred is not None:
        plt.scatter(x, y_pred, c="r", s=4, label="Predictions",)
    plt.legend(prop={"size": 14});


def learning_curve_plotter(train_loss_values, test_loss_values, EPOCH) :
    # plot the learning curves
    plt.plot(np.arange(1, EPOCH+1), train_loss_values, label="Train loss")
    plt.plot(np.arange(1, EPOCH+1), test_loss_values, label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend();

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    '''works only for torch'''
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

