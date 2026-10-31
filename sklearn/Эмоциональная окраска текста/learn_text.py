import pandas as pd
import numpy as np
import re
import nltk
import pyprind
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

stop = stopwords.words('english')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename(columns={'0': 'review', '1': 'sentiment'})
porter = PorterStemmer()
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # скипаем заголовок
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None

    return docs, y

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log_loss', random_state=1)
doc_stream = stream_docs(path='movie_data.csv')

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Точность: {clf.score(X_test, y_test):.3f}')

# обновляем модель
clf = clf.partial_fit(X_test, y_test)

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

# Самые значимые 5 слов
n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Тема {(topic_idx + 1)}:')
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

'''
df['review'] = df['review'].apply(preprocessor)


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
small_param_grid = [
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
        },
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0]
        }
    ]
lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear'))
    ])
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy', cv=5,
                           verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print(f'Лучший набор параметров: {gs_lr_tfidf.best_params_}')
print(f'Точность CV: {gs_lr_tfidf.best_score_:.3f}')
clf = gs_lr_tfidf.best_estimator_
print(f'Точность на тестовом наборе: {clf.score(X_test, y_test):.3f}')
'''
