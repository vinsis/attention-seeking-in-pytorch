import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from loader import loader, sequence_length

# sequence_length is equal to 10
encoder_input_size = 32
encoder_hidden_size = 32

decoder_input_size = 32
decoder_output_size = 32*2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

encoder = nn.LSTM(encoder_input_size, encoder_hidden_size, bidirectional=True, batch_first=True)
embedding = nn.Embedding(num_embeddings=sequence_length, embedding_dim=encoder_input_size)

decoder =nn.LSTM(decoder_input_size, decoder_output_size, batch_first=True)
decoder_output_to_sequence_length = nn.Linear(decoder_output_size, sequence_length)

trainable_parameters = [{'params': net.parameters()} for net in [embedding, encoder, decoder, decoder_output_to_sequence_length]]

optimizer = Adam(trainable_parameters)

embedding.to(device)
encoder.to(device)
decoder.to(device)
decoder_output_to_sequence_length.to(device)

decoder_input = torch.zeros(1, sequence_length, decoder_input_size)

def train():
    for random_sequence in loader:
        random_sequence = random_sequence.to(device)
        correct_sequence = torch.sort(random_sequence)[0]
        correct_sequence = correct_sequence.long().to(device).squeeze(0)
        # correct_sequence = correct_sequence.view(sequence_length, -1)
        # print(correct_sequence)

        random_sequence_embedding = embedding(random_sequence)
        encoder_outputs, (encoder_h, encoder_c) = encoder(random_sequence_embedding)
        encoder_outputs = encoder_outputs[:,:,:encoder_hidden_size] + encoder_outputs[:,:,encoder_hidden_size:]

        # print(encoder_outputs.size())
        decoder_outputs, (decoder_h, decoder_c) = decoder(decoder_input, (encoder_h.view(1,1,-1), encoder_c.view(1,1,-1)))
        softmax_input = decoder_output_to_sequence_length(decoder_outputs).squeeze(0)
        # print(softmax_input.size())
        loss = criterion(softmax_input, correct_sequence)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break

if __name__ == '__main__':
    train()
