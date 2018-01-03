import numpy as np
import pandas as pd



from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

np.random.seed(0)

if __name__ == "__main__":

    #load data
    train_df = pd.read_csv('./train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('./test.tsv', sep='\t', header=0)

    naive_train_df = pd.read_csv('./train.csv', sep=',', header=0, skiprows=1)
    naive_test_df = pd.read_csv('./test.csv', sep=',', header=0, skiprows=1)

    naive_train = naive_train_df.iloc[:,1:2]
    naive_test = naive_train_df.iloc[:,2:3]

    features_train, features_test, target_train, target_test = train_test_split(naive_train, naive_test.values.ravel())

    clf = GaussianNB()
    clf.fit(features_train, target_train)
    target_pred = clf.predict(features_test)
    print("Naive Bayes Accuracy is: "+str(accuracy_score(target_test, target_pred, normalize = True)*100)+"")
    raw_docs_train = train_df['Phrase'].values
    raw_docs_test = test_df['Phrase'].values
    sentiment_train = train_df['Sentiment'].values
    num_labels = len(np.unique(sentiment_train))

    #text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')

    print("pre-processing train docs...")
    processed_docs_train = []
    for doc in raw_docs_train:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_train.append(stemmed)
   
    print ("pre-processing test docs...")
    processed_docs_test = []
    for doc in raw_docs_test:
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_test.append(stemmed)

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

    dictionary = corpora.Dictionary(processed_docs_all)
    dictionary_size = len(dictionary.keys())
    print ("dictionary size: ", dictionary_size) 
    #dictionary.save('dictionary.dict')
    #corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print ("converting to token ids...")
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))
 
    seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)

    #pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)

    #LSTM
    print ("fitting LSTM ...")
    model = Sequential()
    model.add(Embedding(dictionary_size, 128, dropout=0.2))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(word_id_train, y_train_enc, nb_epoch=1, batch_size=256, verbose=1)

    test_pred = model.predict_classes(word_id_test)

    #make a submission
    test_df['Sentiment'] = test_pred.reshape(-1,1) 
    header = ['PhraseId', 'Sentiment']
    test_df.to_csv('./lstm_sentiment.csv', columns=header, index=False, header=True)
