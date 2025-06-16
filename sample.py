from train import category_tensor, input_tensor
from preprocessing import all_letters, n_letters
from model import RNN

import torch

import os

max_length = 20

def sample(category, start_letter='A'):
    with torch.no_grad():
        category_tensor_val = category_tensor(category).to(device)
        input_char_tensor = input_tensor(start_letter).to(device)
        hidden = rnn.initHidden().to(device)

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor_val, input_char_tensor[0], hidden)

            topv, topi = output.topk(1)
            topi = topi[0][0]

            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter

            input_char_tensor = input_tensor(letter).to(device)

        return output_name

def samples(category, start_letters='ABC'):
    print(f"\n--- Sampling for category: {category} ---")
    for start_letter in start_letters:
        generated_name = sample(category, start_letter)
        print(f"{category} ({start_letter}): {generated_name}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = "model.pth"

    if not os.path.exists(model_path):
        print(f"Error: model file not found")
    else:
        rnn = RNN(n_letters, 128, n_letters).to(device)
        rnn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        rnn.eval()
        print("Model loaded successfully!")

        #Run sampling for each category
        samples('Russian', 'RUS')
        samples('German', 'GER')
        samples('Spanish', 'SPA')
        samples('Chinese', 'CHI')
        samples('French', 'ABC')