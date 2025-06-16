import torch
import torch.nn as nn

import time
import math
import random
import pickle

from preprocessing import all_categories, category_lines, n_categories, all_letters, n_letters
from model import RNN

#global variables
criterion = nn.NLLLoss()
learning_rate = 0.005

# helper function get random pairs of (category, line)
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_pair():
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line

def category_tensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS token
    return torch.LongTensor(letter_indexes)

# (category, line) -> (category, input, target)
def random_training_example():
    category, line = random_training_pair()
    cat_tensor = category_tensor(category)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return cat_tensor, input_line_tensor, target_line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# in classification only the last output is used, here calculate loss at every step

def train(cat_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden().to(device)

    rnn.zero_grad()

    loss = torch.Tensor([0]).to(device)

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(cat_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()/input_line_tensor.size(0)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    rnn = RNN(n_letters, 128, n_letters).to(device)

    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0

    start = time.time()

    for iter in range(1, n_iters + 1):
        category_tensor_val, input_line_tensor, target_line_tensor = random_training_example()
        category_tensor_val = category_tensor_val.to(device)
        input_line_tensor = input_line_tensor.to(device)
        target_line_tensor = target_line_tensor.to(device)

        output, loss = train(category_tensor_val, input_line_tensor, target_line_tensor)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    print("Training complete!")

    # save the model and losses for plotting
    torch.save(rnn.state_dict(), "model.pth")
    with open("losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)