import torch.nn as nn
class LanguageEncoder(nn.Module):

    def __init__(self, question_vocab_size, word_embed_size, num_layers, hidden_size, batch_first):
        """
        Extract question featues:
            - step 1: using word2vec
            - step 2: using LSTM
        """
        super(LanguageEncoder, self).__init__()
        self.word2vec = nn.Embedding(question_vocab_size, word_embed_size)
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers = num_layers,  batch_first =  batch_first)

    def forward(self, question):

        question_vec = self.word2vec(question)
        _, (hidden, cell) = self.lstm(question_vec) 
        
        return hidden[0]