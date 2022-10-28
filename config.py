data_cfg = {
    "data_dir": "/home/ngoc/data/WSDM2023/",
    "batch_size":16,
    "embeded_size": 1024,
    "word_embeded_size": 300,
    "max_question_length": 50,
    "image_size": (512,512),
    "test_running": False
}

model_cfg = {
    "model_name": "SAN",
    "num_epochs": 30,
    "num_fc_layers": 1,
    "hidden_fc_size": 1024,
    "learning_rate": 0.001,
    "decay_step_size": 10,
    "gamma": 0.1,
    "max_question_length": 50,
    "rnn_batch_first": True,
    "attention_size": 1024,
    "num_att_layers": 2,
    "checkpoint_dir": "saved_model/",
    "wandb_log": True,
    "saved_image": "outputs/",
    "pre_train": False,
    "train": True
} 