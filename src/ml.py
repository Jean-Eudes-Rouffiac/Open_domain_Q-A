import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Masking
from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout


def create_model(embedding_matrix, FLAGS):
    embedding = Embedding(
        *embedding_matrix.shape, 
        weights=[embedding_matrix], 
        trainable=False, 
        mask_zero=True
    )
    
    q_in = Input(shape=(None,))
    q = embedding(q_in)
    q = SpatialDropout1D(0.2)(q)
    q = Bidirectional(LSTM(100, return_sequences=True))(q)
    q = GlobalMaxPooling1D()(q)
    
    
    t_in = Input(shape=(None,))
    t = embedding(t_in)
    t = SpatialDropout1D(0.2)(t)
    t = Bidirectional(LSTM(150, return_sequences=True))(t)
    t = GlobalMaxPooling1D()(t)
    
    hidden = concatenate([q, t])
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.5)(hidden)
    
    out1 = Dense(1, activation='sigmoid')(hidden)
    
    model = Model(inputs=[t_in, q_in], outputs=out1)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def train_reader(train_text, train_questions, train_targets, embedding, FLAGS):
    model = create_model(embedding, FLAGS)
    train_history = model.fit(
    [train_text, train_questions], 
    train_targets,
    epochs=2,
    validation_split=0.2,
    batch_size=64
    )
    model.save('%s/reader_model.h5'%FLAGS.model_folder)
    
def error_summary(model, train_text, train_questions, train_targets, test_text, test_questions, test_targets, train_reader, test_reader, FLAGS):
    print(train_reader.columns)
    model.summary()
    pred_train_target = model.predict([train_text, train_questions], batch_size=32)
    pred_test_target = model.predict([test_text, test_questions], batch_size=32)
    
    train_reader['pred_target'] = pred_train_target
    test_reader['pred_target'] = pred_test_target
    
    train_reader = train_reader.groupby(['annotation_id'], sort=False)['pred_target'].max()
    test_reader = test_reader.groupby(['annotation_id'], sort=False)['pred_target'].max()

    print(train_reader.columns)
    print(train_reader)
    
    train_accuracy = 100 * train_reader['is_long_answer'].sum() / len(train_reader)
    test_accuracy = 100 * test_reader['is_long_answer'].sum() / len(test_reader)
    
    
    return train_accuracy, test_accuracy