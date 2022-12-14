from dataset import NSMC_Dataset
from trainer import return_training_arguments
from metrics import compute_metrics, sentences_predict
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

import torch

def main():
    train_dataset = NSMC_Dataset('train')
    test_dataset = NSMC_Dataset('test')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    training_args = return_training_arguments()

    MODEL_NAME = "bert-base-multilingual-cased"
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    train_trainer = Trainer(
    model=model,                         # the instantiated π€ Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    )
    # print('νμ΅μμ.. 1 epochμ λλ΅ 30λΆ μ λ μμλ©λλ€ :-)')
    # train_trainer.train() # 1 epochμ λλ΅ 30λΆ μ λ μμλ©λλ€ :-)


    eval_trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
    )
    # print('evaluate μμ..' )
    # eval_trainer.evaluate(eval_dataset=test_dataset)


    pred_param = [model,train_dataset.tokenizer,device]
    print(sentences_predict(["μν κ°μ¬λ°μ΄ γγγγγ"]+pred_param))
    # print(sentences_predict("μ§μ§ μ¬λ―Έμλ€μ γγ"))
    # print(sentences_predict("λ λλ¬Έμ μ§μ§ μ§μ¦λ"))
    # print(sentences_predict("μ λ§ μ¬λ°κ³  μ’μμ΄μ."))


    
    
if __name__ == '__main__':
    main()