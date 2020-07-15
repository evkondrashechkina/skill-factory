import numpy as np
import pandas as pd
import math
import re
regex = re.compile("[,\.!?:;‘’0-9_$@{}']")

SPAM = 'SPAM'
NOT_SPAM = 'NOT_SPAM'
train_data = [  
    ['Купите новое чистящее средство', SPAM],   
    ['Купи мою новую книгу', SPAM],  
    ['Подари себе новый телефон', SPAM],
    ['Добро пожаловать и купите новый телевизор', SPAM],
    ['Привет давно не виделись', NOT_SPAM], 
    ['Довезем до аэропорта из пригорода всего за 399 рублей', SPAM], 
    ['Добро пожаловать в Мой Круг', NOT_SPAM],  
    ['Я все еще жду документы', NOT_SPAM],  
    ['Приглашаем на конференцию Data Science', NOT_SPAM],
    ['Потерял твой телефон напомни', NOT_SPAM],
    ['Порадуй своего питомца новым костюмом', SPAM]
]
df_ru = pd.DataFrame(train_data,columns = ['email','label'])

df_en = pd.read_csv('spam_or_not_spam.csv',header=0)
df_en['label'] = df_en['label'].apply(lambda x: 'NOT_SPAM' if x==0 else 'SPAM')
df_en = df_en.dropna()

class SpamClassificator():
    def __init__(self,dataset_language='ru'):#dataset_language=['ru','en']
        self.pA = 0
        self.pNotA = 0
        self.dictSpam = {}
        self.dictNotSpam = {}
        self.allUniqueWords = []
        self.dataset_language = dataset_language
        
    def get_probabilities(self):
        return (self.pA,self.pNotA)
    
    def get_words(self,text):
        transformed_text = regex.sub(' ', text).lower()
        words = list(filter(None,transformed_text.split()))
        return words
    
    def calculate_word_frequencies(self, body, label='SPAM'):#'SPAM','NOT_SPAM'
        words = self.get_words(body)
        for word in words:
            if label == 'SPAM':
                if word in self.dictSpam.keys():
                    self.dictSpam[word] = self.dictSpam[word] + 1
                else:
                    self.dictSpam[word] = 1
            else:
                if word in self.dictNotSpam.keys():
                    self.dictNotSpam[word] = self.dictNotSpam[word] + 1
                else:
                    self.dictNotSpam[word] = 1
            if word not in self.allUniqueWords:
                self.allUniqueWords.append(word)
    
    def train(self):
        print('Training...')
        countA = 0
        countNotA = 0
        if self.dataset_language == 'ru':
            df_train = df_ru
        else:
            df_train = df_en
        for date,email in df_train.iterrows():
            if email[1] == 'SPAM':
                countA = countA + 1
            else:
                countNotA = countNotA + 1
            self.calculate_word_frequencies(email[0],email[1]) 
        self.pA = countA / (countA + countNotA)
        self.pNotA = countNotA / (countA + countNotA)
    
    def calculate_P_Bi_A(self,word, label):
        if label == 'SPAM':
            #+1 - смещение по Лапласу для всех слов
            if word in self.dictSpam.keys():
                word_spam_count = self.dictSpam[word]+1
            else:
                word_spam_count = 1
            #http://bazhenov.me/blog/2012/06/11/naive-bayes.html - это почему добавляем len(self.allUniqueWords)
            return np.longdouble(word_spam_count / (sum(self.dictSpam.values())+len(self.allUniqueWords)))
        else:
            if word in self.dictNotSpam.keys():
                word_not_spam_count = self.dictNotSpam[word]+1
            else:
                word_not_spam_count = 1
            return np.longdouble(word_not_spam_count / (sum(self.dictNotSpam.values())+len(self.allUniqueWords)))
    
    def calculate_P_B_A(self,text, label):
        words = self.get_words(text)
        p = 0
        for word in words:
            p = p+math.log(self.calculate_P_Bi_A(word,label))
        return p
                                 
    def classify(self,email):
        p_spam = self.calculate_P_B_A(email,'SPAM')+math.log(self.pA)
        p_not_spam = self.calculate_P_B_A(email,'NOT_SPAM')+math.log(self.pNotA)
        if p_spam>p_not_spam:
            return True
        else:
            return False
