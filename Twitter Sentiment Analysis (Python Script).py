# Recurrent Neural Network for Twitter Sentiment Analysis #

#import necessary modules
import os
import re 
import math 
import random
import string
import numpy as np
import pandas as pd 
import seaborn as sns
import tensorflow as tf   
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import layers, optimizers, regularizers  
from tensorflow.keras.callbacks import EarlyStopping, Callback 
from tensorflow.keras.preprocessing.text import Tokenizer      
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Adjust pandas data display settings 
pd.set_option('display.max_colwidth', 100)

#Set plotting context and style
sns.set_context('notebook')
sns.set_style('white')

#Set random seed for reproducible results
rs = 121

#set global random seed to libraries used 
random.seed(rs)
np.random.seed(rs)
tf.random.set_seed(rs)


#Helper Functions for data analysis and visualization
#Defining a function to compute and report error scores
def error_scores(ytest, ypred, model_accuracy, classes):
    error_metrics = {
        'Accuracy': model_accuracy,
        'Precision': precision_score(ytest, ypred, average=None),
        'Recall': recall_score(ytest, ypred, average=None),
        'F1 score': f1_score(ytest, ypred, average=None),
    }

    return pd.DataFrame(error_metrics, index=classes).apply(lambda x:round(x,2)).T

#Define function to plot the confusion matrix using a heatmap
def plot_cm(cm, labels):
    plt.figure(figsize=(10,7))
    hmap = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
    hmap.set_xlabel('Predicted Value', fontsize=13)
    hmap.set_ylabel('Truth Value', fontsize=13)
    plt.tight_layout()

#Define custom function to visualize model training history
def plot_training_history(run_histories: list, metrics: list = None, title='Model run history'):
    #If no specific metrics are given, infer them from the first history object
    if not metrics:
        metrics = [key for key in run_histories[0].history.keys() if 'val_' not in key]
    else:
        metrics = [metric.lower() for metric in metrics]

    #Set up the number of rows and columns for the subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)  #Limit to a max of 3 columns for better readability
    n_rows = math.ceil(n_metrics / n_cols)

    #Set up colors to use
    colors = ['steelblue', 'red', 'skyblue', 'orange', 'indigo', 'green', 'DarkCyan', 'olive', 'brown', 'hotpink']

    #Ensure loss first is plotted first
    if 'loss' in metrics:
        metrics.remove('loss')
        metrics.insert(0,'loss')

    #Initialize the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5*n_cols, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    #Loop over each metric and create separate subplots
    for i, metric in enumerate(metrics):
        #Initialize starting epoch
        epoch_start = 0
        for j, history in enumerate(run_histories):
            epochs_range = range(epoch_start, epoch_start + len(history.epoch))

            #Plot training and validation metrics for each run history
            axes[i].plot(epochs_range, history.history[metric], color=colors[i*2], ls='-', lw=2, label=(f'Training {metric}') if j==0 else None)
            if f'val_{metric}' in history.history:
                axes[i].plot(epochs_range, history.history.get(f'val_{metric}', []), color=colors[i*2+1], ls='-', lw=2, label=(f'Validation {metric}') if j==0 else None)

            #Update the epoch start for the next run
            epoch_start += len(history.epoch)

        #Set the titles, labels, and legends
        axes[i].set(title=f'{metric.capitalize()} over Epochs', xlabel='Epoch', ylabel=metric.capitalize())
        axes[i].legend(loc='best')

    #Remove any extra subplots if the grid is larger than the number of metrics
    for k in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[k])

    fig.suptitle(title, fontsize=16, y=(0.95) if n_rows>1 else 0.98)
    plt.show()

#Define custom function decode tokens, returning them to raw text
def decode_tokens(indexed_tokens, idx2word_dict):
  return ' '.join([idx2word_dict[index] for index in indexed_tokens if index!=0])


#Part One: Reading and Inspecting the Data
#Access and read data into dataframe
df = pd.read_csv(r'/Users/mmd96/Desktop/My Folders/Python/Completed Projects/16. Recurrent Neural Network for Sentiment Analysis/Tweets.csv')

#Report total count 
print(f'Total number of tweets: {df.shape[0]:,}')
print()

#Inspecting the data
#Preview a sample
print(df.sample(10))
print()

#Inspect columns, data types, number of non-null entries
print(df.info())
print()

#Get statistical overview of the data 
print(df.describe().T)
print()

#Drop null entries 
df = df.dropna(ignore_index=True)

#report number of empty rows after dropping 
print('Number of empty rows:', df.isnull().sum().sum())
print()


                                    ###################


#Part Two: Data Preparation and Preprocessing
#Label Encoding 
#Perform label encoding on the target class
classes = {'neutral': 0, 'positive': 1, 'negative': 2}

#Replace string labels with numeric labels
df['sentiment'] = df['sentiment'].replace(classes)

#Examine Class Distribution 
#We can check the distribution of classes for the target variable, 'sentiment'
print('Class Distribution (in %):\n')
print(df['sentiment'].value_counts(normalize=True).apply(lambda x: f'{x*100:.2f}%'),'\n\n')

#Visualizing the class distribution in the data using count plot
plt.figure(figsize=(10,7))
ax = sns.countplot(x=df['sentiment'], hue=df['sentiment'], order=[2,0,1], hue_order=[2,0,1], saturation=.6, palette='Set1')
ax.set_title('Class Distribution', fontsize=20, pad=14)
ax.set_xlabel('Sentiment', fontsize=14)
ax.set_xticklabels(['negative', 'neutral', 'positive'])
ax.legend(labels=['negative', 'neutral', 'positive'])
plt.show()


#Text Preprocessing 
#Create custom function to clean text. This function should normalize the text, remove hyperlinks, 
# remove mentions and hashtags, remove stop words, and lemmatize the text 
#Instantiate nltk's lemmatizer and stop words' list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#Preprocessing function
def preprocess_text(text):
    text = text.lower().strip()  #lowercase and remove whitespaces
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  #remove URLs 
    text = re.sub(r'@\w+|#\w+', '', text)  #remove mentions and hashtags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  #remove punctuations
    text = text.split()   #split string to list
    text = [word for word in text if word not in stop_words]  #remove stop words 
    text = [lemmatizer.lemmatize(word) for word in text]   #lemmatize text (nouns)
    return ' '.join(text)

#Define a function to remove emojis and emoticons from text
def remove_emojis(text):
    #Regex pattern to match emojis and emoticons
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  #emoticons
        u"\U0001F300-\U0001F5FF"  #symbols & pictographs
        u"\U0001F680-\U0001F6FF"  #transport & map symbols
        u"\U0001F700-\U0001F77F"  #alchemical symbols
        u"\U0001F1E0-\U0001F1FF"  #flags (iOS)
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)  #remove emojis


#Preprocess all tweets and save into new column
df['text_preprocessed'] = df['text'].apply(preprocess_text).replace('', np.nan)

#Drop empty rows
df = df.dropna(ignore_index=True)

#Apply second function to remove all symbols and emojis 
df['text_preprocessed'] = df['text_preprocessed'].apply(remove_emojis)

#preview a sample after preprocessing
print('Sample of preprocessed text:\n')
print(df['text_preprocessed'].sample(5))
print()



#Text Tokenization
#Instantiate tokenizer 
tokenizer = Tokenizer()

#tokenize text corpus 
tokenizer.fit_on_texts(df['text_preprocessed'])

#convert the text into sequences of word indice
df['text_preprocessed'] = tokenizer.texts_to_sequences(df['text_preprocessed'])

#get token indices and report vocabulary size
word2idx = tokenizer.word_index
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx) + 1
print('vocabulary size:', vocab_size)

#Sequence padding
#Get maximum sequence length
max_seq_len = max([len(seq) for seq in df['text_preprocessed']])

#apply padding
df['text_preprocessed'] = list(pad_sequences(df['text_preprocessed'], padding='post', maxlen=max_seq_len))

#Preview data sample
print(df.sample(5))
print()


#Data Selection
#Identify predictor and target variables
X_data = df['text_preprocessed']
y_data = df['sentiment'].values


#Stratified Data Splitting 
#Obtain training, testing, and validation sets (70% training / 15% testing / 15% validation)
#first split 70/30
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, train_size=0.7, stratify=y_data, random_state=rs)
#second split 67/33
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, stratify=y_temp,  random_state=rs)

#Convert data to numpy arrays
X_train = tf.convert_to_tensor(X_train.tolist(), dtype=tf.int32)
X_val = tf.convert_to_tensor(X_val.tolist(), dtype=tf.int32)
X_test = tf.convert_to_tensor(X_test.tolist(), dtype=tf.int32)

#Check the sizes of the training, validation and testing sets
print(f'Number of training samples: {X_train.shape[0]:,}')    
print(f'Number of validation samples: {X_val.shape[0]:,}')    
print(f'Number of testing samples: {X_test.shape[0]:,}')
print()


                                    ###################


#Part Three: Model Development and Evaluation 
#In this section, I will develop, train and evaluate a recurrent neural network for the present 
# task of sentiment analysis.    
#Establishing a Performance baseline 
#instantiate a logistic regression object
LR = LogisticRegression(max_iter=500, random_state=rs)

#fit the model
model = LR.fit(X_train, y_train)

#generate predictions
y_pred = model.predict(X_test)

#report error scores 
print('Logistic Regression classification results:')
print(error_scores(y_test, y_pred, accuracy_score(y_test, y_pred), classes=classes))
print()


#Pre-training Preparations
#Preparing a Sentiment Lexicon for Sentiment Masking (using EmoLex)
#Load NRC Emotion Lexicon into a dictionary
emotional_words_set = set()
with open("NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", "r") as f:
    for line in f:
        word, emotion, association = line.strip().split("\t")
        if int(association) == 1:
            emotional_words_set.add(word)

#sort and obtain final list
emotional_words_lst = list(sorted(emotional_words_set))

#Convert emotional words to their indices (if found)    
sentiments_vocab_indices = [word2idx[word] for word in emotional_words_lst if word in word2idx]

#Preview a sample 
print(np.random.choice(emotional_words_lst, 10))
print()


#Preparing Embeddings using GloVe 
#Define embeddings dimensions 
embedding_dims = 300 

#Create embeddings matrix using GloVe 
#build embeddings index from the GloVe text file
embeddings_index = {}
glove_path = r'/Users/mmd96/Desktop/My Folders/Python/Completed Projects/15. Recurrent Neural Network-Based Recommender System/glove.840B.300d.txt'
with open(glove_path, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector_values = values[1:]
        if len(vector_values) > embedding_dims:
            vector_values = vector_values[-embedding_dims:]
        coefs = np.asarray(vector_values, dtype='float32')
        embeddings_index[word] = coefs

#Create embedding matrix 
embedding_matrix = np.zeros((vocab_size, embedding_dims))
for word, idx in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector



#MODEL DEVELOPMENT
#Define sentiment mask layer 
class SentimentMaskLayer(layers.Layer):
    def __init__(self, sentiments_vocab_indices, sentiment_weighing_factor=1.0, **kwargs):
        super(SentimentMaskLayer, self).__init__(**kwargs)
        #Initialize parameters
        self.sentiment_vocab_tensor = tf.constant([idx for idx in sentiments_vocab_indices], dtype=tf.int32)
        self.sentiment_weighing_factor = sentiment_weighing_factor

    def call(self, inputs):
        #Inputs 
        embedding_outputs, text_tokens = inputs 

        #Compare all tokens with sentiment words
        sentiment_matches = tf.reduce_any(tf.equal(tf.expand_dims(text_tokens, -1), self.sentiment_vocab_tensor), axis=-1)  #Shape: (batch_size, seq_len)

        #Apply the sentiment weighting factor where matches are found 
        sentiment_mask = tf.cast(sentiment_matches, tf.float32) * self.sentiment_weighing_factor
        sentiment_mask = tf.expand_dims(sentiment_mask, -1)   # Shape: (batch_size, seq_len, 1)

        #Assign weight importances: Multiply by (1 + sentiment_mask) 
        return tf.cast(embedding_outputs * (1.0 + sentiment_mask), dtype=tf.float32)

#Define self-attention layer 
class SelfAttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        #Initialize weights for query, key, and value matrices
        dims = input_shape[-1]   #input dimensions
        self.WQ = self.add_weight(shape=(dims, dims), initializer='glorot_uniform', trainable=True)
        self.WK = self.add_weight(shape=(dims, dims), initializer='glorot_uniform', trainable=True)
        self.WV = self.add_weight(shape=(dims, dims), initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        #Compute query, key, and value matrices
        Q = tf.matmul(inputs, self.WQ)
        K = tf.matmul(inputs, self.WK)
        V = tf.matmul(inputs, self.WV)

        #Compute key matrix dimensions for scaling attention scores
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)

        #Compute attention scores
        attention_scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)

        #Compute attention weights 
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        #Multipy attention weights with value matrix to get attention output 
        attention_output = tf.matmul(attention_weights, V)  # shape: (batch_size, seq_len, features)
        return attention_output


#Create Keras model subclass to build a RNN model 
class RNN_Network(tf.keras.Model):    
    def __init__(self, output_dims, embedding_input=50000, embedding_dims=300, LSTM_units=256,
                 Conv1D_filters=128, sentiments_vocab_indices=None, sentiment_weighing_factor=1.0, **kwargs):
        super().__init__(name='RNN_Network', **kwargs)

        #Define model layers 
        #Embedding layer and dropout
        self.Embedding_layer = layers.Embedding(input_dim=embedding_input, output_dim=embedding_dims, embeddings_regularizer=regularizers.l2(0.0001), 
                                                mask_zero=True, trainable=True, weights=[embedding_matrix], name='Embedding_layer')
        self.Dropout1 = layers.Dropout(0.5, name='Dropout_layer1')
        
        #Gaussian noise layer
        self.Noise_layer = layers.GaussianNoise(0.08, name='Gaussian_Noise_layer')

        #Sentiment mask and dropout 
        self.SentimentMaskLayer = SentimentMaskLayer(sentiments_vocab_indices, sentiment_weighing_factor=sentiment_weighing_factor, name='Sentiment_Mask_layer')
        self.Dropout2 = layers.Dropout(0.5, name='Dropout_layer2')

        #Self-Attention layer and dropout
        self.Attention_layer = SelfAttentionLayer(name='Self-Attention_layer')
        self.Dropout3 = layers.Dropout(0.5, name='Dropout_layer3')
        
        #Convolutional and pooling layers (for varying n-grams processing)
        self.Conv1D_layer1 = layers.Conv1D(filters=Conv1D_filters, kernel_size=3, padding='same', activation='relu', name='Conv1D_3grams_layer')
        self.MaxPool_layer1 = layers.MaxPooling1D(name='MaxPool_layer1')

        self.Conv1D_layer2 = layers.Conv1D(filters=Conv1D_filters, kernel_size=4, padding='same', activation='relu', name='Conv1D_4grams_layer')
        self.MaxPool_layer2 = layers.MaxPooling1D(name='MaxPool_layer2')

        #Concatenation and Reshaping layers 
        self.Concatenate_layer = layers.Concatenate(axis=-1, name='Concatenate_layer')
        self.Reshape_layer = layers.Reshape((1, -1), name='Reshape_layer') 

        #Bidirectional LSTM layer and spatial dropout
        self.Bidirectional_LSTM_layer = layers.Bidirectional(
            layers.LSTM(units=LSTM_units, activation='tanh', return_sequences=True,
                        kernel_regularizer=regularizers.l2(0.001), name='LSTM_layer'), 
                    name='Bidirectional_LSTM_layer')
        self.SpatialDropout = layers.SpatialDropout1D(0.5, name='SpatialDropout_layer')

        #Global Average Pooling layer
        self.GlobalAvgPool_layer = layers.GlobalAveragePooling1D(name='GlobalAvgPooling1D_layer')
        
        #Final classification layer
        self.Classification_layer = layers.Dense(output_dims, activation='softmax', kernel_regularizer=regularizers.l2(0.001), name='Classification_layer')


    def call(self, inputs, training=None):
        text_tokens = inputs 
        #Text Embedding
        text_embeddings = self.Embedding_layer(inputs)
        text_embeddings = self.Dropout1(text_embeddings)

        #Apply Gaussian Noise
        if training:
            text_embeddings = self.Noise_layer(text_embeddings, training=training)

        #Sentiment Masking         
        embeddings_masked = self.SentimentMaskLayer([text_embeddings, text_tokens])
        embeddings_masked = self.Dropout2(embeddings_masked)

        #Self-attention
        attention_output = self.Attention_layer(embeddings_masked)
        attention_output = self.Dropout3(attention_output)

        #Convolutional layers for n-grams
        threegrams_features = self.Conv1D_layer1(attention_output)
        threegrams_features = self.MaxPool_layer1(threegrams_features)
        fourgrams_features = self.Conv1D_layer2(attention_output)
        fourgrams_features = self.MaxPool_layer2(fourgrams_features)

        #Merge and reshape features 
        merged_features = self.Concatenate_layer([threegrams_features, fourgrams_features])
        features_reshaped = self.Reshape_layer(merged_features)

        #Bidirectional LSTM         
        bi_LSTM_output = self.Bidirectional_LSTM_layer(features_reshaped)
        bi_LSTM_output = self.SpatialDropout(bi_LSTM_output)
        
        #Global average pooling 
        global_avg_output = self.GlobalAvgPool_layer(bi_LSTM_output)
        
        #Final classification 
        final_outputs = self.Classification_layer(global_avg_output)
        return final_outputs


#Instantiating the RNN model
#Build RNN network using model subclass
RNN_model = RNN_Network(embedding_input=vocab_size,
                        LSTM_units=128,
                        Conv1D_filters=128,
                        output_dims=len(np.unique(y_data)),
                        sentiments_vocab_indices=sentiments_vocab_indices,
                        sentiment_weighing_factor=2.0)


#Training configurations 
#Model Compilation
#Compile the model
RNN_model.compile(optimizer=optimizers.Adam(learning_rate=0.0002, epsilon=1e-6), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#Learning Rate Schedule and Early Stopping
#Build a custom Adaptive Learning Rate for the optimizer
class AdaptiveLearningRate(Callback):
    '''
    Custom learning rate scheduler that implements an adaptive learning rate strategy for 
    the optimizer during model training.
    '''
    def __init__(self, metric='val_loss', higher_is_better=False, patience=5, decrease_factor=0.5, 
                 min_lr=0.000001, min_delta=0.0, start_from_epoch=0, use_absolute_best=True, verbose=1):
        super(AdaptiveLearningRate, self).__init__()
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.patience = patience
        self.decrease_factor = decrease_factor
        self.min_lr = min_lr
        self.min_delta = min_delta 
        self.last_best_score = -np.inf if higher_is_better else np.inf
        self.use_absolute_best = use_absolute_best
        self.start_from_epoch = max(start_from_epoch - 1, 0)   #to keep up with a count start of 1 (instead of 0)
        self.verbose = verbose
        self.wait = 0


    def on_epoch_end(self, epoch, logs=None):
        #Get current metric value (loss or any other metric)
        current_score = logs.get(self.metric)

        if current_score is None:
            return

        if epoch >= self.start_from_epoch:
            #Check for improvement
            improvement = (
                    ((current_score - self.last_best_score) > self.min_delta) 
                    if self.higher_is_better
                    else ((self.last_best_score - current_score) > self.min_delta)
                )

            if improvement:
                self.last_best_score = current_score
                self.wait = 0  #Reset wait since improvement happened
            else:
                self.wait += 1

            #Check if patience is exceeded, reduce the learning rate
            if self.wait >= self.patience:
                #Reduce learning rate by a decrease factor and ensure it does not go below min_lr
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = max(current_lr * self.decrease_factor, self.min_lr)
                self.model.optimizer.learning_rate.assign(new_lr)   #Set new learning rate
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: Learning rate reduced to {new_lr:.5f}.")
                if not self.use_absolute_best:    #uses last best instead of the absolute best
                    self.last_best_score = current_score   
                self.wait = 0   #Reset wait after learning rate adjustment


#Adaptive learning rate scheduler 
lr_scheduler = AdaptiveLearningRate(metric='val_loss', patience=2, decrease_factor=0.5, min_lr=0.00001, min_delta=0.01, start_from_epoch=5)

#Define early stopping criterion
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, start_from_epoch=10)


#Model Training and Evaluation 
#Fit the model (30 training epochs)
RNN_run_history = RNN_model.fit(X_train, y_train, 
                                epochs=30,
                                batch_size=16,
                                validation_data=(X_val, y_val),
                                callbacks=[lr_scheduler, early_stop])

#Visualize run history
plot_training_history([RNN_run_history], metrics=['loss', 'accuracy'])


#Model Evaluation
#Evaluate the model on the testing set
loss, accuracy = RNN_model.evaluate(X_test, y_test, verbose=0)

#Get class predictions
y_pred = RNN_model.predict(X_test, verbose=0).argmax(axis=-1)

#Report results
print('Model Evaluation Results:\n')
print(error_scores(y_test, y_pred, accuracy, classes=classes))
print()


#Generating Sentiment Predictions from a Random Sample
#Extracting a random sample from the dataset
random_indices = tf.constant(np.random.choice(len(X_test), size=10, replace=False), dtype=tf.int32)
X_sample = tf.gather(X_test, random_indices)

#Generate sentiment predictions using the model
predicted_sentiment = RNN_model.predict(X_sample).argmax(axis=-1)

#Decode tokens of selected sample 
X_sample_raw = [decode_tokens(row, idx2word) for row in X_sample.numpy()]

#Create a dataframe
X_sample_df = pd.DataFrame({'Tweet (preprocessed)': X_sample_raw, 
                            'Predicted Sentiment': predicted_sentiment})

X_sample_df['Predicted Sentiment'] = X_sample_df['Predicted Sentiment'].map({0: 'neutral', 1: 'positive', 2: 'negative'})

#Display results 
print(X_sample_df)