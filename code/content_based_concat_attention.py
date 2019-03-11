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
content_based_attention_concat = nn.Linear(encoder_hidden_size*4, encoder_hidden_size*2)
reference_vector = torch.Tensor(1,1,encoder_hidden_size*2).uniform_().requires_grad_(True)

trainable_parameters = [{'params': net.parameters()} for net in [embedding, encoder, decoder, decoder_output_to_sequence_length, content_based_attention_concat]]
trainable_parameters += [{'params': reference_vector}]

optimizer = Adam(trainable_parameters, lr=0.001)

embedding.to(device)
encoder.to(device)
decoder.to(device)
decoder_output_to_sequence_length.to(device)
content_based_attention_concat.to(device)

decoder_input = torch.zeros(1, 1, decoder_input_size)

def train():
    correct, total = 0, 0
    for index, random_sequence in enumerate(loader):
        random_sequence = random_sequence.to(device)
        correct_sequence = torch.sort(random_sequence)[1]
        correct_sequence = correct_sequence.long().to(device).squeeze(0)

        random_sequence_embedding = embedding(random_sequence)
        encoder_outputs, (encoder_h, encoder_c) = encoder(random_sequence_embedding)

        decoder_outputs = []
        decoder_input_h = encoder_h.view(1,1,-1)
        for time in range(sequence_length):
            # attention starts here
            decoder_input_h_repeated = decoder_input_h.repeat(1,sequence_length,1)
            concatenated_tensor = torch.cat([decoder_input_h_repeated, encoder_outputs], dim=2)
            transformed_concatenated_tensor = content_based_attention_concat(concatenated_tensor)
            similarity_with_reference_vector = torch.bmm(reference_vector, transformed_concatenated_tensor.transpose(1,2))
            encoder_output_weights = F.softmax(similarity_with_reference_vector, dim=2)
            weighted_sum_of_encoder_outputs = torch.bmm(encoder_output_weights, encoder_outputs)
            # attention ends here
            decoder_output_at_time_t, (decoder_h, decoder_c) = decoder(decoder_input, (decoder_input_h, weighted_sum_of_encoder_outputs))
            decoder_outputs.append(decoder_output_at_time_t)
            decoder_input_h = decoder_h

        decoder_outputs = torch.cat(decoder_outputs, 1)
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

def test():
    with torch.no_grad():
        for index, random_sequence in enumerate(loader):
            random_sequence = random_sequence.to(device)
            correct_sequence = torch.sort(random_sequence)[1]
            correct_sequence = correct_sequence.long().to(device).squeeze(0)

            random_sequence_embedding = embedding(random_sequence)
            encoder_outputs, (encoder_h, encoder_c) = encoder(random_sequence_embedding)

            decoder_outputs = []
            decoder_input_h = encoder_h.view(1,1,-1)
            attentions = []
            for time in range(sequence_length):
                # attention starts here
                decoder_input_h_repeated = decoder_input_h.repeat(1,sequence_length,1)
                concatenated_tensor = torch.cat([decoder_input_h_repeated, encoder_outputs], dim=2)
                transformed_concatenated_tensor = content_based_attention_concat(concatenated_tensor)
                similarity_with_reference_vector = torch.bmm(reference_vector, transformed_concatenated_tensor.transpose(1,2))
                encoder_output_weights = F.softmax(similarity_with_reference_vector, dim=2)
                weighted_sum_of_encoder_outputs = torch.bmm(encoder_output_weights, encoder_outputs)
                attentions.append(encoder_output_weights)
                # attention ends here
                decoder_output_at_time_t, (decoder_h, decoder_c) = decoder(decoder_input, (decoder_input_h, weighted_sum_of_encoder_outputs))
                decoder_outputs.append(decoder_output_at_time_t)
                decoder_input_h = decoder_h

            decoder_outputs = torch.cat(decoder_outputs, 1)
            softmax_input = decoder_output_to_sequence_length(decoder_outputs).squeeze(0)

            loss = criterion(softmax_input, correct_sequence)
            accurate = (softmax_input.max(1)[1] == correct_sequence).sum()
            return random_sequence, correct_sequence, softmax_input, accurate, attentions

if __name__ == '__main__':
    train()
