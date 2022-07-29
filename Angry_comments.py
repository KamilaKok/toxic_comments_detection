# TODO разработать бинарный классификатор
# TODO максимизировать RECALL
# TODO PRECISION > 0.95

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')

data_list = pd.read_csv('./data/labeled.csv', sep=',')

data_list['toxic'] = data_list['toxic'].apply(int)
# .apply - к каждой строчке применяем одну и ту же функцию(приведение float к int)
train_data_list, test_data_list = train_test_split(data_list, test_size=500)

sentence_example = data_list.iloc[1]['comment']
snowball = SnowballStemmer(language='russian')
russian_stop_words = stopwords.words('russian')


def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language='russian')  # разбиваем на токены
    tokens = [i for i in tokens if i not in string.punctuation]  # удаляем знаки пунктуации
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]  # удаляем стоп-слова
    tokens = [snowball.stem(i) for i in tokens]  # приводим к нижнему регистру и удаляем окончания
    return tokens


vectorizer = TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))
features = vectorizer.fit_transform(train_data_list['comment'])
model = LogisticRegression(random_state=0)
model.fit(features, train_data_list['toxic'])

# model_pipline = Pipeline([
#     ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
#     ('model', LogisticRegression(random_state=0))
# ]
# )
#
# model_pipline.fit(train_data_list['comment'], train_data_list['toxic'])
# print(model_pipline.fit(train_data_list['comment'], train_data_list['toxic']))


# prec, rec, thresholds = precision_recall_curve(y_true=test_data_list['toxic'],
#                                                probas_pred=model_pipline.predict_proba(test_data_list['comment'])[:, 1])
#
# disp = PrecisionRecallDisplay.from_estimator(estimator=model_pipline, X=test_data_list['comment'],
#                                              y=test_data_list['toxic'])
# disp.plot()
# plt.show()
# array_np = np.where(prec > 0.95)
#
# print(array_np)
# print(precision_score(y_true=test_data_list['toxic'],
#                       y_pred=model_pipline.predict_proba(test_data_list['comment'])[:, 1] > thresholds[444]))
# print(recall_score(y_true=test_data_list['toxic'],
#                    y_pred=model_pipline.predict_proba(test_data_list['comment'])[:, 1] > thresholds[444]))


model_pipline_c_10 = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
    ('model', LogisticRegression(random_state=0, C=10))
]
)
model_pipline_c_10.fit(train_data_list['comment'], train_data_list['toxic'])

prec_c_10, rec_c_10, thresholds_c_10 = precision_recall_curve(y_true=test_data_list['toxic'],
                                                              probas_pred=model_pipline_c_10.predict_proba(
                                                                  test_data_list['comment'])[:, 1])

print(precision_score(y_true=test_data_list['toxic'],
                      y_pred=model_pipline_c_10.predict_proba(test_data_list['comment'])[:, 1] > thresholds_c_10[444]))
print(recall_score(y_true=test_data_list['toxic'],
                   y_pred=model_pipline_c_10.predict_proba(test_data_list['comment'])[:, 1] > thresholds_c_10[444]))

disp_c_10 = PrecisionRecallDisplay.from_estimator(estimator=model_pipline_c_10, X=test_data_list['comment'],
                                                  y=test_data_list['toxic'])
disp_c_10.plot()
plt.show()
array_np = np.where(prec_c_10 > 0.95)
print(array_np)

# grid_pipline = Pipeline([
#     ('vectorizer', TfidfVectorizer(tokenizer=lambda x: tokenize_sentence(x, remove_stop_words=True))),
#     ('model',
#      GridSearchCV(
#          LogisticRegression(random_state=0),
#          param_grid={'C': [0.1, 1, 10]},
#          cv=3,
#          verbose=4
#      )
#      )
# ])


text = ""
while text != '0':
    text = input('Введите комментарий: ')
    check = model_pipline_c_10.predict([text])
    if 1 in check:
        print('Это токсичный комментарий')
    else:
        print('Это нейтральный комментарий')
