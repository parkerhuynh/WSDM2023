from config import data_cfg, model_cfg
from data_loaders import data_loaders
from models.models import SANModel
import torch
from torchsummary import summary
from train import train
from utils import Text_Dict 
def main():
    print("Start")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloaders = data_loaders(
        data_dir = data_cfg["data_dir"],
        max_question_length = data_cfg["max_question_length"],
        batch_size = data_cfg["batch_size"],
        image_size = data_cfg["image_size"],
        test_running = data_cfg["test_running"]
        )
    question_vocab_size = dataloaders['train'].dataset.question_dict.vocab_size
    
    
    if model_cfg["model_name"] == "SAN":
        model = SANModel(
        embed_size = data_cfg["embeded_size"],
        question_vocab_size = question_vocab_size,
        word_embed_size = data_cfg["word_embeded_size"],
        num_layers= model_cfg["num_fc_layers"],
        hidden_size= model_cfg["hidden_fc_size"],
        batch_first = model_cfg["rnn_batch_first"],
        att_ff_size = model_cfg["attention_size"],
        num_att_layers = model_cfg['num_att_layers']).to(device)
    print(summary(model))
    print(model)
    if model_cfg["train"]:
        print("- Start Training")
        train(model, dataloaders["train"], dataloaders["val"], device)
    

    


if __name__ == "__main__":
    main()