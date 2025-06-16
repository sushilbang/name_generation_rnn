import matplotlib.pyplot as plt
import pickle
import os

if __name__ == "__main__":
    losses_file = "losses.pkl"

    if not os.path.exists(losses_file):
        raise FileNotFoundError(f"Losses file '{losses_file}' not found. Please run the training script first.")
    else:
        with open(losses_file, "rb") as f:
            all_losses = pickle.load(f)

        plt.figure(figsize=(10, 5))
        plt.plot(all_losses)
        plt.title('Training Loss')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.grid(True)

        plot_filename = "training_loss.png"
        plt.savefig(plot_filename)
        plt.show()
        plt.close()