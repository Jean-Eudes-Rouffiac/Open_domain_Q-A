# coding: utf8
import click
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import json
import sqlite3
from tensorflow.keras.models import load_model
from src.utils import dump_pickle, load_pickle
from src.data_processing import load_data, data_processing, count_vocabulary, analyze_vocabulary, get_tf_idf, get_top_n_candidates, pre_process_for_model_reader
from src.ml import train_reader,error_summary

sys.path.append('./src')


def apply_load_data(file_path, FLAGS):
    load_data(file_path, FLAGS)


def apply_retriever_processing(data, FLAGS):
    data = data_processing(data, 'text', FLAGS)
    dump_pickle(data,'%s/data_retriever_processing.pickle'%FLAGS.data_folder)


def apply_text_analyze(data,FLAGS):
    count_vocabulary(data, FLAGS)
    word_count = load_pickle('%s/word_counter.pickle'%FLAGS.data_folder)
    analyze_vocabulary(word_count, FLAGS)


def apply_document_question_similarity(data, FLAGS):
    data = get_tf_idf(data, FLAGS)
    dump_pickle(data,'%s/data_retriever_tf_idf.pickle'%FLAGS.data_folder)
    data = get_top_n_candidates(data, 5, FLAGS)
    dump_pickle(data, '%s/data_retriever_candidates.pickle'%FLAGS.data_folder)


def apply_retriever_metrics(data, FLAGS):
        data['index1'] = data.index
        data['result'] = data[['top_candidates','index1']].apply(lambda x: 1 if x['top_candidates'][0] == x['index1'] else 0 , axis=1)
        data['result2'] = data[['top_candidates','index1']].apply(lambda x: 1 if x['index1'] in x['top_candidates'] else 0 , axis=1)
        score_exact = 100 * data['result'].sum() / len(data)
        score_topn = 100 * data['result2'].sum() / len(data)
        result_retriever_file_path = '%s/retriever_perf.csv' % (FLAGS.results_folder)
        with open(result_retriever_file_path, 'w') as dump_file:
            dump_file.write("Score exact;Score top 5\n")
            dump_file.write("%.2f;%.2f\n" % (score_exact, score_topn))


def apply_reader_processing(data, FLAGS):
    train_text, train_questions, test_text, test_questions, train_targets, test_targets, embedding= pre_process_for_model_reader(data, 'is_long_answer', FLAGS)
    dump_pickle(train_text, '%s/train_text.pickle'%FLAGS.data_folder)
    dump_pickle(train_questions, '%s/train_questions.pickle'%FLAGS.data_folder)
    dump_pickle(test_text, '%s/test_text.pickle'%FLAGS.data_folder)
    dump_pickle(test_questions, '%s/test_questions.pickle'%FLAGS.data_folder)
    dump_pickle(train_targets, '%s/train_targets.pickle'%FLAGS.data_folder)
    dump_pickle(test_targets, '%s/test_targets.pickle'%FLAGS.data_folder)
    dump_pickle(embedding, '%s/embedding.pickle'%FLAGS.data_folder)


def apply_train_reader(FLAGS):
    train_text = load_pickle('%s/train_text.pickle'%FLAGS.data_folder)
    train_questions = load_pickle('%s/train_questions.pickle'%FLAGS.data_folder)
    train_targets = load_pickle('%s/train_targets.pickle'%FLAGS.data_folder)
    embedding = load_pickle('%s/embedding.pickle'%FLAGS.data_folder)
    train_reader(train_text, train_questions, train_targets, embedding, FLAGS)

def apply_metrics_reader(FLAGS):
    data_train = load_pickle('%s/train_reader.pickle'%FLAGS.data_folder)
    data_test = load_pickle('%s/test_reader.pickle'%FLAGS.data_folder)
    train_text = load_pickle('%s/train_text.pickle'%FLAGS.data_folder)
    train_questions = load_pickle('%s/train_questions.pickle'%FLAGS.data_folder)
    train_targets = load_pickle('%s/train_targets.pickle'%FLAGS.data_folder)
    test_text = load_pickle('%s/test_text.pickle'%FLAGS.data_folder)
    test_questions = load_pickle('%s/test_questions.pickle'%FLAGS.data_folder)
    test_targets = load_pickle('%s/test_targets.pickle'%FLAGS.data_folder)
    train_reader = load_pickle('%s/train_reader.pickle'%FLAGS.data_folder)
    test_reader = load_pickle('%s/test_reader.pickle'%FLAGS.data_folder)
    model = load_model('%s/reader_model.h5'%FLAGS.model_folder)
    train_accuracy, test_accuracy = error_summary(model, train_text, train_questions, train_targets, test_text, test_questions, test_targets, train_reader, test_reader, FLAGS)
    result_reader_file_path = '%s/reader_perf.csv' % (FLAGS.results_folder)
    with open(result_reader_file_path, 'w') as dump_file:
        dump_file.write("Train accuracy;Test accuracy\n")
        dump_file.write("%.2f;%.2f\n" % (train_accuracy, test_accuracy))


def run():

# Define step
    tf.compat.v1.app.flags.DEFINE_string('step', '', 'process step')

# Define directory folders
    tf.compat.v1.app.flags.DEFINE_string('data_folder', 'data/', 'data folder')
    tf.compat.v1.app.flags.DEFINE_string('results_folder', 'results/', 'results folder')
    tf.compat.v1.app.flags.DEFINE_string('model_folder', 'model/', 'model folder')

# Define size of train data set
    tf.compat.v1.app.flags.DEFINE_integer('n_rows_data_set', 2000, 'size of data set')

    FLAGS = tf.compat.v1.app.flags.FLAGS



    if FLAGS.step == "load_data":
        path = '%s/simplified-nq-train.jsonl' %(FLAGS.data_folder)
        apply_load_data(path, FLAGS)

    if FLAGS.step == "retriever_processing":
        data = load_pickle('%s/data_retriever.pickle' %(FLAGS.data_folder))
        apply_retriever_processing(data, FLAGS)

    if FLAGS.step == "analyze_text":
        data = load_pickle('%s/data_retriever_processing.pickle' %(FLAGS.data_folder))
        apply_text_analyze(data, FLAGS)

    if FLAGS.step == "document_question_similarity":
        data = load_pickle('%s/data_retriever_processing.pickle' %(FLAGS.data_folder))
        apply_document_question_similarity(data, FLAGS)

    if FLAGS.step == "retriever_metrics":
        data = load_pickle('data/data_retriever_candidates.pickle')
        apply_retriever_metrics(data, FLAGS)

    if FLAGS.step == "reader_processing":
        data = load_pickle('%s/data_reader.pickle'%FLAGS.data_folder)
        apply_reader_processing(data, FLAGS)

    if FLAGS.step == "train_reader":
        apply_train_reader(FLAGS)

    if FLAGS.step == "reader_metrics":
        apply_metrics_reader(FLAGS)



    if FLAGS.step == 'test':
        data = load_pickle('%s/data_candidates.pickle'%FLAGS.data_folder)
        print(data.columns)
        print(data[['question_text', 'question_text_tokens', 'top_candidates']].head(10))
        #data[['question_text', 'question_text_tokens', 'top_candidates']].to_excel('data.xlsx')




# Run the main
if __name__ == '__main__':
    run()
