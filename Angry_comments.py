# TODO разработать бинарный классификатор
# TODO максимизировать RECALL
# TODO PRECISION > 0.95

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import plot_precision_recall_curve
import numpy as np
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')

data_list = pd.read_csv('./data/labeled.csv', sep=',')

data_list['toxic'] = data_list['toxic'].apply(int)
# .apply - к каждой строчке применяем одну и ту же функцию(приведение float к int)

# print(data_list.shape)
# print(data_list.head(5))
# print(data_list['toxic'].value_counts())
# print(*(i for i in data_list[data_list['toxic'] == 1]['comment'].head(6)))
# print(*(i for i in data_list[data_list['toxic'] == 0]['comment'].head(5)))

train_data_list, test_data_list = train_test_split(data_list, test_size=500)
# print(test_data_list.shape)
# print(test_data_list['toxic'].value_counts())
# print(train_data_list['toxic'].value_counts())


sentence_example = data_list.iloc[1]['comment']
snowball = SnowballStemmer(language='russian')
russian_stop_words = stopwords.words('russian')


def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence_example, language='russian')  # разбиваем на токены
    tokens = [i for i in tokens if i not in string.punctuation]  # удаляем знаки пунктуации
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]  # удаляем стоп-слова
    tokens = [snowball.stem(i) for i in tokens]  # приводим к нижнему регистру и удаляем окончания
    return tokens


model_pipline = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ('model', LogisticRegression(random_state=0))
]
)

model_pipline.fit(train_data_list['comment'], train_data_list['toxic'])
# print(model_pipline.fit(train_data_list['comment'], train_data_list['toxic']))
# text = 1
# while text != 0:
#     text = input('Введите комментарий: ')
# print(model_pipline.predict(['хуйло']))
#
# print(precision_score(y_true=test_data_list['toxic'], y_pred=model_pipline.predict(test_data_list['comment'])))
# print(recall_score(y_true=test_data_list['toxic'], y_pred=model_pipline.predict(test_data_list['comment'])))
