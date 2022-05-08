# CS6910 Assignment 3
 CS6910 Fundamentals of Deep Learning.

Team members: Dip Narayan Gupta(CS21Z025),Monica (CS21Z023)


1. To train a Seq2Seq model for Dakshina Dataset transliteration from English to Hindi, use the notebook: **seq2seqvanilla.ipynb**.
  
2. To train a Seq2Seq model with Attention for Dakshina Dataset transliteration from English to Hindi, use the notebook: **attention_with_wandb and attention_without_wandb**.

### General Framework:

ALl our notebooks have been created in Google Colab with a GPU backend. We have used TensorFlow and Keras for defining, training and testing our model.

### Vanilla Seq2Seq model:


# Loading dataset
 
def load_data(path):

Returns important information about the data like input characters, target characters



# Getting unique tokens
 
hindi_tokens , english_tokens = unique_tokenize(train)

Returns hindi_tokens , english token

# Mapping the tokens
 
def tokenize_map(hindi_tokens , english_tokens)

Returns maapping for each tokens


# Preprocessing the datset 


def process(data):

# Generating encoder decoder models for LSTM , RNN, GRU

def build_model(cell = "LSTM",units = 32, enc_layers = 1, dec_layers = 1,embedding_dim = 32,dense_size=32,dropout=None):

# Defines a vanilla encoder-decoder model using the following hyperparameters: 

units: Number of cells in the encoder and decoder layers

cell_type: choice of cell type: Simple RNN, LSTM, GRU

enc_layers: Number of layers in the encoder

dec_layers: Number of layers in the decoder

input_embedding_size: Dimenions of the vector to represent each character

dropout_fraction: fraction of neurons to drop out


`
Train_with_wandb()

 Trains, validates the model on the data and logs the accuracies and losses into wandb.
The characterwise validation accuracy with teacher forcing is logged per epoch. The inference validation accuracy without teacher forcing is logged after the complete training phase. change the what procedure we used validation accuary 

def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'learning_rate': 1e-2,
        'dense_size': 128,
        'units': 128,
        'cell': 'LSTM',
        'embedding_dim': 64,
        'enc_layers': 1,
        'dec_layers': 1,
        'dropout': 0.,
        'batch_size': 64
    }

# ModelInitialisation
train,enc,dec = build_model(units=256,dense_size=512,enc_layers=2,dec_layers=3,cell = "GRU", embedding_dim = 64)
# Early Stopping 
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

# To save the model with best validation accuracy
checkpoint = ModelCheckpoint('bestmodel.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

# fit the data 
train.fit([trainx,trainxx],trainy,
         batch_size=128,
         validation_data=([valx,valxx],valy),
         epochs=10,
          callbacks=[WandbCallback(), earlyStopping,checkpoint])
          
# Prepare Inference model:         
def inference(inp,dec_layers,cell="LSTM"):
Takes in a model that has the cell_type and converts into an inference model. ie it reorders the connections of a model

`
# Getting test accuracy
def test_accuracy(pred): 

Returns acuuracy for test data which are predicted correctly


**prediction_vanilla.csv**: 
This file contains the prediction with highest score made using decoder  for the entire test set. It contains the original English word, the reference Hindi word and the pred_vanilla predicted by model.



### Attention Seq2Seq model:

# Loading dataset
 
def load_data(path):

Returns important information about the data like input characters, target characters



# Getting unique tokens
 
hindi_tokens , english_tokens = unique_tokenize(train)

Returns hindi_tokens , english token

# Mapping the tokens
 
def tokenize_map(hindi_tokens , english_tokens)

Returns maapping for each tokens



# Defining AttentionLayer

class Attention(tf.keras.layers.Layer):

This class implements Bahdanau attention and creates a layer called attention that can be integrated with keras very easily


# Generating encoder decoder models for LSTM , RNN, GRU

def build_model(cell = "LSTM",units = 32, enc_layers = 1, dec_layers = 1,embedding_dim = 32,dense_size=32,dropout=None):

Defines a vanilla encoder-decoder model using the following hyperparameters: 

units: Number of cells in the encoder and decoder layers

cell_type: choice of cell type: Simple RNN, LSTM, GRU

enc_layers: Number of layers in the encoder

dec_layers: Number of layers in the decoder

input_embedding_size: Dimenions of the vector to represent each character

dropout_fraction: fraction of neurons to drop out



# Train_with_wandb()

Trains, validates the model on the data and logs the accuracies and losses into wandb.
The characterwise validation accuracy with teacher forcing is logged per epoch. The inference validation accuracy without teacher forcing is logged after the complete training phase.


def train():
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        'learning_rate': 1e-2,
        'dense_size': 128,
        'units': 128,
        'cell': 'LSTM',
        'embedding_dim': 64,
        'enc_layers': 1,
        'dec_layers': 1,
        'dropout': 0.,
        'batch_size': 64
    }


Trains and validates the model on the data. The characterwise validation accuracy with teacher forcing is plotted per epoch. The inference validation accuracy without teacher forcing is printed after the complete training phase.


# Model Initilisation
train,enc,dec = build_model(units=256,dense_size=512,enc_layers=2,dec_layers=3,cell = "GRU", embedding_dim = 64)

# fit the data 
train.fit([trainx,trainxx],trainy,
         batch_size=128,
         validation_data=([valx,valxx],valy),
         epochs=10,
          callbacks = [checkpoint])
          
# Prepare inference model        
def inference(inp,dec_layers,cell="LSTM"):
Takes in a model that has the cell_type  and converts into an inference model. ie it reorders the connections of a model

# Beam search 
def beam_search(inp,k,dec_layers,cell="LSTM"):

# prediction for beam search 
def test_accuracy_beam(prediction):
 returns accuracy based on top k words


# Plot heat mapping
def plot_attention(attention, sentence, predicted_sentence,orig,hind,deco):

Generate heatmap for given input and predicted sentences`


**VesperLibre-Regular.ttf**: Font used for printing Hindi characters in plots.

Visualise words in form of connectivity as gif using
visualise(which_word) '
**prediction_attention.csv**: This file contains the original English word, the reference Hindi word, and predicted first word using beam search.


