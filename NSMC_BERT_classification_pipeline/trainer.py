from transformers import BertForSequenceClassification, Trainer, TrainingArguments


def return_training_arguments():
    training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=1,              # total number of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=500,
            save_steps=500,
            save_total_limit=2
        )

    return training_args

