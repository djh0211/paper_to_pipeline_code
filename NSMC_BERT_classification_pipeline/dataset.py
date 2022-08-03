import datasets
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, BertTokenizer

class NSMC_Dataset(Dataset):
    def __init__(self, type):
        assert type in ['train', 'test'], 'train/test 중 입력해주세요.'
            
        self.dataset = datasets.load_dataset('nsmc')
        self.type = type
    # 필요한 데이터인 document와 label 정보만 pandas라이브러리 DataFrame 형식으로 변환
        self.train_data, self.test_data = self.make_df()
        self.MODEL_NAME = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.tokenized_train_sentences = self.tokenizer(
            list(self.train_data['document']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            )

        self.tokenized_test_sentences = self.tokenizer(
            list(self.test_data['document']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            )

        self.train_label = self.train_data['label'].values
        self.test_label = self.test_data['label'].values

        print()
        print(self.type + ' dataset 생성 완료.....')

    def make_df(self):
            train_data = pd.DataFrame({"document":self.dataset['train']['document'], "label":self.dataset['train']['label'],})
            test_data = pd.DataFrame({"document":self.dataset['test']['document'], "label":self.dataset['test']['label'],})
    # 데이터 중복을 제외한 갯수 확인
            print("학습데이터 : ",train_data['document'].nunique()," 라벨 : ",train_data['label'].nunique())
            print("데스트 데이터 : ",test_data['document'].nunique()," 라벨 : ",test_data['label'].nunique())

            # 중복 데이터 제거
            train_data.drop_duplicates(subset=['document'], inplace= True)
            test_data.drop_duplicates(subset=['document'], inplace= True)

            # 데이터셋 갯수 확인
            print('중복 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
            print('중복 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))

            # null 데이터 제거
            train_data['document'].replace('', np.nan, inplace=True)
            test_data['document'].replace('', np.nan, inplace=True)
            train_data = train_data.dropna(how = 'any')
            test_data = test_data.dropna(how = 'any')

            print('null 제거 후 학습 데이터셋 : {}'.format(len(train_data)))
            print('null 제거 후 테스트 데이터셋 : {}'.format(len(test_data)))

            return train_data, test_data


    def __getitem__(self, idx):
        # pass
        if self.type == 'train':
            item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_train_sentences.items()}
            item['labels'] = torch.tensor(self.train_label[idx])
            return item
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_test_sentences.items()}
            item['labels'] = torch.tensor(self.test_label[idx])
            return item

    def __len__(self):
        # pass
        if self.type == 'train':
            return len(self.train_label)
        else:
            return len(self.test_label)

    
    





