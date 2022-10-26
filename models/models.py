from models.visual_extraction import VisualEncoder
from models.language_extraction import LanguageEncoder
from models.attention import Attention
import torch.nn as nn

class SANModel(nn.Module):

    def __init__(self, embed_size, question_vocab_size, word_embed_size, num_layers, hidden_size, batch_first, att_ff_size, num_att_layers):
        """
        Fusing Image feature and question feature using Full Connected Layer
        """
        super(SANModel, self).__init__()
        self.image_encoder = VisualEncoder(embed_size)
        self.question_encoder = LanguageEncoder(question_vocab_size, word_embed_size,  num_layers, hidden_size,batch_first)
        
        self.san = nn.ModuleList(
            [Attention(input_embed_size = embed_size,  num_channels=att_ff_size)] * num_att_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(embed_size,embed_size),
            nn.Linear(embed_size,embed_size),
            nn.Linear(embed_size,4))
    def forward(self, image, question):

        image_feature = self.image_encoder(image)
        question_feature = self.question_encoder(question)
        vi = image_feature
        u = question_feature
        for att_layer in self.san:
            u = att_layer(vi, u)
        output = self.mlp(u) 
        return  output