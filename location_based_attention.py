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

embedding = nn.Embedding(num_embeddings=sequence_length, embedding_dim=encoder_input_size)
encoder = nn.LSTM(encoder_input_size, encoder_hidden_size, bidirectional=True, batch_first=True)

decoder =nn.LSTM(decoder_input_size, decoder_output_size, batch_first=True)
decoder_output_to_sequence_length = nn.Linear(decoder_output_size, sequence_length)

# Based on technique mentioned in right column of page 3 of http://aclweb.org/anthology/D15-1166
# We only use encoder_h and replace encoder_c with weighted sum of encoder_outputs
location_based_attention = nn.Linear(encoder_hidden_size*2, sequence_length)

trainable_parameters = [{'params': net.parameters()} for net in [embedding, encoder, decoder, decoder_output_to_sequence_length, location_based_attention]]

optimizer = Adam(trainable_parameters, lr=0.005)

embedding.to(device)
encoder.to(device)
decoder.to(device)
decoder_output_to_sequence_length.to(device)
location_based_attention.to(device)

decoder_input = torch.zeros(1, sequence_length, decoder_input_size)

def train():
    correct, total = 0, 0
    for index, random_sequence in enumerate(loader):
        random_sequence = random_sequence.to(device)
        correct_sequence = torch.sort(random_sequence)[1]
        correct_sequence = correct_sequence.long().to(device).squeeze(0)

        random_sequence_embedding = embedding(random_sequence)
        encoder_outputs, (encoder_h, encoder_c) = encoder(random_sequence_embedding)

        # attention starts here
        encoder_output_weights = F.softmax(location_based_attention(encoder_h.view(1,1,-1)), dim=2)
        weighted_sum_of_encoder_outputs = torch.bmm(encoder_output_weights, encoder_outputs)
        # attention ends here

        decoder_outputs, (decoder_h, decoder_c) = decoder(decoder_input, (encoder_h.view(1,1,-1), weighted_sum_of_encoder_outputs))
        softmax_input = decoder_output_to_sequence_length(decoder_outputs).squeeze(0)

        loss = criterion(softmax_input, correct_sequence)

        # calculating accuracy
        accurate = (softmax_input.max(1)[1] == correct_sequence).sum()
        correct += accurate
        total += sequence_length
        if index%100 == 0:
            print('Loss at iteration {}: {:.8f}'.format(index, loss.item()))
            print('Accuracy in last 100 iterations: {}/{}'.format(correct, total))
            correct, total = 0, 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break

if __name__ == '__main__':
    train()
