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
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    )
    # print('학습시작.. 1 epoch에 대략 30분 정도 소요됩니다 :-)')
    # train_trainer.train() # 1 epoch에 대략 30분 정도 소요됩니다 :-)


    eval_trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
    )
    # print('evaluate 시작..' )
    # eval_trainer.evaluate(eval_dataset=test_dataset)


    pred_param = [model,train_dataset.tokenizer,device]
    print(sentences_predict(["영화 개재밌어 ㅋㅋㅋㅋㅋ"]+pred_param))
    # print(sentences_predict("진짜 재미없네요 ㅋㅋ"))
    # print(sentences_predict("너 때문에 진짜 짜증나"))
    # print(sentences_predict("정말 재밌고 좋았어요."))


    
    
if __name__ == '__main__':
    main()