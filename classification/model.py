import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from classification.utils import get_sequences_lengths, variable, argmax


class Seq2SeqModelAttention(torch.nn.Module):
    def __init__(self,  hidden_size, padding_idx,init_idx,max_len,
                 vocab_size, embedding_dim, embeddings=None):
        """
        Sequence-to-sequence model
        :param vocab_size: the size of the vocabulary
        :param embedding_dim: Dimension of the embeddings
        :param hidden_size: The size of the encoder and the decoder
        :param padding_idx: Index of the special pad token
        :param init_idx: Index of the <s> token
        :param max_len: Maximum length of a sentence in tokens
        :param teacher_forcing: Probability of teacher forcing
        """
        super().__init__()

        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.combined_hidden_size = hidden_size * 2
        self.padding_idx = padding_idx
        self.init_idx = init_idx
        self.max_len = max_len
        # self.teacher_forcing = teacher_forcing
        self.vocab_size = vocab_size
        self.output_size = 1

        self.drop = nn.Dropout(p=0.5)

        self.attn = nn.Linear(self.combined_hidden_size + self.embedding_dim,
                              self.max_len)
        self.attn_combine = nn.Linear(self.combined_hidden_size +self.embedding_dim,
                                      self.combined_hidden_size)

        if self.embeddings is not None:
            self.emb = nn.Embedding.from_pretrained(
                self.embeddings, freeze=False)
        else:
            self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.enc = nn.LSTM(self.embedding_dim, self.hidden_size,
                           batch_first=True,
                           bidirectional= True)
        self.dec = nn.LSTMCell(self.combined_hidden_size,
                               self.combined_hidden_size)
        self.lin = nn.Linear(self.combined_hidden_size, self.vocab_size)

        self.M = nn.Linear(self.combined_hidden_size, self.combined_hidden_size)

        self.bias = nn.parameter(torch.zeros(1))

    def unfreeze_embeddings(self):
        self.emb = nn.Embedding.from_pretrained(
            self.embeddings, freeze=False)

    def zero_state(self, batch_size):
        """
        Create an initial hidden state of zeros
        :param batch_size: the batch size
        :return: A tuple of two tensors (h and c) of zeros of the shape of (batch_size x hidden_size)
        """

        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        h0 = variable(torch.zeros(state_shape))
        c0 = variable(torch.zeros(state_shape))

        return h0, c0

    def encode_sentence(self, inputs):
        """
        Encode input sentences input a batch of hidden vectors z
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x hidden_size)
        """
        batch_size = inputs.size(0)

        # Get lengths
        lengths = get_sequences_lengths(inputs, masking=self.padding_idx)

        # Sort as required for pack_padded_sequence input
        lengths, indices = torch.sort(lengths, descending=True)
        inputs = inputs[indices]

        lengths = lengths.data.tolist()

        # Pack
        inputs = torch.nn.utils.rnn.pack_padded_sequence(
            self.drop(self.emb(inputs)), lengths, batch_first=True)


        # Encode
        hidden, cell = self.zero_state(batch_size)
        output, (hidden, cell) = self.enc(inputs, (hidden, cell))

        output = torch.nn.utils.rnn.pad_packed_sequence(
            output, total_length=self.max_len)[0]

        hidden = torch.cat((hidden[0], hidden[1]), dim=1)

        _, unsort_ind = torch.sort(indices)
        z = output[:,unsort_ind]
        z1 = hidden[unsort_ind]

        return z.view(batch_size, self.max_len, self.combined_hidden_size), z1

    def decoder_state(self, z):
        """
        Create initial hidden state for the decoder based on the hidden vectors z
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tuple of two tensors (h and c) of size (batch_size x hidden_size)
        """

        batch_size = z.size(0)

        state_shape = (batch_size, self.combined_hidden_size)

        c0 = variable(torch.zeros(state_shape))
        return z, c0

    def decoder_initial_inputs(self, batch_size):
        """
        Create initial input the decoder on the first timestep
        :param inputs: The size of the batch
        :return: A vector of size (batch_size, ) filled with the index of self.init_idx
        """

        inputs = variable(torch.LongTensor(batch_size, ).fill_(self.init_idx))

        return inputs

    def decode_sentence(self, encoder_outputs, decoder_hidden, targets=None):
        """
        Decode the tranlation of the input sentences based in the hidden vectors z and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x hidden_size)
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        batch_size = encoder_outputs.size(0)
        # c is the prev
        encoder_outputs, decoder_state = self.decoder_state(encoder_outputs)

        x_i = self.decoder_initial_inputs(batch_size)

        outputs = []
        for i in range(self.max_len):
            embedded = self.drop(self.emb(x_i).view(1, batch_size,-1))

            attn_weights = self.attn(torch.cat((embedded[0], decoder_state), 1))

            attn_weights = F.softmax(attn_weights, dim=1)

            attn_weights = attn_weights.view(batch_size, 1, self.max_len)

            attn_applied = torch.bmm(attn_weights, encoder_outputs).view(
                batch_size, self.combined_hidden_size)

            output = torch.cat((embedded[0], attn_applied), 1)
            output = self.attn_combine(output)

            decoder_hidden, decoder_state = self.dec(
                output, (decoder_hidden, decoder_state))

            output = self.lin(decoder_hidden)
            if targets is not None and i < len(targets):
                x_i = targets[:, i]
            else:
                x_i = torch.multinomial(F.softmax(output, dim=-1), 1).squeeze(-1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def auto_encoder(self, inputs, targets = None, unfreeze = False):
        """
                Perform the forward pass of the network and return
                non-normalized probabilities of the output tokens at each
                timestep
                :param inputs: A tensor of size (batch_size x max_len) of
                indices of input sentences' tokens
                :return: A tensor of size (batch_size x max_len x vocab_size)
                """

        # if self.training and np.random.rand() < self.teacher_forcing:
        #     targets = inputs
        # else:
        #     targets = None

        z, z1 = self.encode_sentence(inputs)
        outputs = self.decode_sentence(z, z1, targets)
        return outputs


    def forward(self, sentence_1, sentence_2, targets=None):
        """
        Perform the forward pass of the network and return non-normalized probabilities of the output tokens at each timestep
        :param inputs: A tensor of size (batch_size x max_len) of indices of input sentences' tokens
        :return: A tensor of size (batch_size x max_len x vocab_size)
        """

        _, z_1 = self.encode_sentence(sentence_1)
        _, z_2 = self.encode_sentence(sentence_2)

        outputs = self.classify(torch.cat((z_1, z_2), dim=1))
        negative = variable(torch.ones(outputs.size()))
        negative = negative - outputs
        outputs = torch.stack((outputs, negative), dim=-2)
        return outputs