import matplotlib.pyplot as plt

def plot_metrics(losses, accuracies):
    plt.figure()

    plt.plot(losses, label="Loss")
    plt.plot(accuracies, label="Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Performance")
    plt.legend()

    plt.savefig("metrics.png")
    plt.close()