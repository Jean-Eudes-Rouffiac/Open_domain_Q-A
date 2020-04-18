import json
import gc
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import collections
import re
import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from src.utils import dump_pickle, load_pickle
from sklearn.model_selection import train_test_split
import fasttext

nltk.download('stopwords')


def load_data(file_path, FLAGS):
    sampling_rate=15
    reader_lines = []
    retriever_lines = []

    with open(file_path) as f:
        for i in enumerate(range(FLAGS.n_rows_data_set)):
            line = f.readline()
            if not line:
                break

            line = json.loads(line)

            reader_lines.append({
                'text': line['document_text'],
                'question': line['question_text']
            })

            text = line['document_text'].split(' ')
            question = line['question_text']
            annotations = line['annotations'][0]

            for i, candidate in enumerate(line['long_answer_candidates']):
                label = i == annotations['long_answer']['candidate_index']

                start = candidate['start_token']
                end = candidate['end_token']

                if label or (i % sampling_rate == 0):
                    retriever_lines.append({
                        'text': " ".join(text[start:end]),
                        'is_long_answer': label,
                        'question': question,
                        'annotation_id': annotations['annotation_id']
                    })

    data_retriever = pd.DataFrame(reader_lines).fillna(-1)
    dump_pickle(data_retriever, '%s/data_retriever.pickle' %(FLAGS.data_folder))
    data_reader = pd.DataFrame(retriever_lines)
    dump_pickle(data_reader, '%s/data_reader.pickle' %(FLAGS.data_folder))



def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = raw_html.replace(cleanr, '')
    return cleantext

def remove_ponctation(object_column):
    ponctuation_list = ['!','#','$','%','&','(',')','*','+',',','.','/',':',';','<','=','>','?','@','[','\'',']','^','_','{','|','}','~','\\', "'"]
    for ponctuation in ponctuation_list:
        object_column = object_column.apply(lambda x: x.replace(ponctuation, ""))
    return object_column


def data_processing(data, column, FLAGS):
    stemmer= PorterStemmer()
    stop_words = set(stopwords.words('english'))
    data[column+'_tokens'] = data[column].apply(lambda x: str(x))
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: str.lower(x))
    data[column+'_tokens'] = cleanhtml(data[column+'_tokens'])
    data[column+'_tokens'] = remove_ponctation(data[column+'_tokens'])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: x.replace('-', " "))
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: x.split(" "))
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [mot for mot in x if mot])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [mot for mot in x  if (mot not in stop_words)])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [mot for mot in x  if (mot != '``')])
    data[column+'_tokens'] = data[column+'_tokens'].apply(lambda x: [stemmer.stem(mot) for mot in x])
    return data

def count_vocabulary(data, FLAGS):
    word_counts = {}
    for tokens in data['document_text_tokens']:
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
    word_counter = collections.Counter(word_counts)
    dump_pickle(word_counter, '%s/word_counter.pickle'%FLAGS.data_folder)


def analyze_vocabulary(word_counter, FLAGS):
    fig = plt.figure(figsize=(12, 5))
    lst = word_counter.most_common(30)
    df = pd.DataFrame(lst, columns = ['Word', 'Count'])
    fig = df.plot.bar(x='Word',y='Count').get_figure()
    fig.savefig('%s/mots.png'%FLAGS.data_folder)
    print(word_counter.most_common(100))


def get_tf_idf(data, FLAGS):
    data['text_tokens'] = data['text_tokens'].apply(lambda x: ' '.join(x))
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(data['text_tokens'])
    dump_pickle(x.toarray(), '%s/tf_idf_matrix.pickle'%FLAGS.data_folder)
    dump_pickle(tfidf,'%s/tf_idf.pickle'%FLAGS.data_folder)
    return data


def get_top_n_candidates(data, n, FLAGS):
    x = load_pickle('%s/tf_idf_matrix.pickle'%FLAGS.data_folder)
    tfidf = load_pickle('%s/tf_idf.pickle'%FLAGS.data_folder)
    question = data_processing(data, 'question', FLAGS)
    data['question_tokens'] = data['question_tokens'].apply(lambda x: ' '.join(x))
    top_candidates = []
    i = 0
    for question in data['question_tokens'] :
        print(question)
        i+= 1
        print(i)
        question_tfidf = tfidf.transform([question])
        similarity = cosine_similarity(x, question_tfidf.toarray())
        top_n_idx = heapq.nlargest(n, range(len(similarity)), similarity.take)
        top_candidates.append(top_n_idx)
    data['top_candidates'] = top_candidates
    return data



def split_features_targets(data, target_column):
    return data.drop(target_column, axis=1), data[target_column].astype(int).values


def pre_process_for_model_reader(data, target_column, FLAGS):
    train, test = train_test_split(data, test_size=0.20, random_state=1)
    training_features, training_targets = split_features_targets(train, target_column)
    test_features, test_targets = split_features_targets(test, target_column)

    dump_pickle(train, '%s/train_reader.pickle'%FLAGS.data_folder)
    dump_pickle(test, '%s/test_reader.pickle'%FLAGS.data_folder)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False, num_words=80000)
    for text in [training_features['text'], test_features['text'], training_features['question'], test_features['question']]:
        tokenizer.fit_on_texts(text.values)

    train_text = tokenizer.texts_to_sequences(training_features['text'].values)
    train_questions = tokenizer.texts_to_sequences(training_features['question'].values)
    test_text = tokenizer.texts_to_sequences(test_features['text'].values)
    test_questions = tokenizer.texts_to_sequences(test_features['question'].values)

    train_text = sequence.pad_sequences(train_text, maxlen=300)
    train_questions = sequence.pad_sequences(train_questions)
    test_text = sequence.pad_sequences(test_text, maxlen=300)
    test_questions = sequence.pad_sequences(test_questions)

    embedding = np.zeros((tokenizer.num_words + 1, 300))

    pre_train_fasttext = fasttext.load_model('%s/wiki.en.bin'%FLAGS.data_folder)

    for word, i in tokenizer.word_index.items():
        if i >= tokenizer.num_words - 1:
            break
        embedding[i] = pre_train_fasttext.get_word_vector(word)

    dump_pickle(tokenizer, '%s/tokenizer.pickle'%FLAGS.model_folder)
    return train_text, train_questions, test_text, test_questions, training_targets, test_targets, embedding
