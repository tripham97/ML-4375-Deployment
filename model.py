import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import zipfile

# Define embedding dimension and latent dimension
embedding_dim = 256
latent_dim = 512

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

# Extract the contents of the ZIP file
with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
    zip_ref.extractall()

# Read the content of the translation file
path_to_file = 'spa-eng/spa.txt'
df = pd.read_csv(path_to_file, sep='\t', header=None, names=['source', 'target'])

# Modify the code to use the first 100,000 entries
subset_df = df.head(20000)

# Split the data into training and testing sets
train_df, test_df = train_test_split(subset_df, test_size=0.1, random_state=42)

# Tokenize the source and target texts
source_tokenizer = Tokenizer(filters='')
source_tokenizer.fit_on_texts(train_df['source'])
target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(train_df['target'])

# Add <start>, <end>, and <unknown> tokens to the vocabulary
start_token = '<start>'
end_token = '<end>'
unknown_token = '<unknown>'
target_tokenizer.word_index[start_token] = len(target_tokenizer.word_index) + 1
target_tokenizer.word_index[end_token] = len(target_tokenizer.word_index) + 2
target_tokenizer.word_index[unknown_token] = len(target_tokenizer.word_index) + 3
target_tokenizer.index_word[len(target_tokenizer.word_index) + 1] = start_token
target_tokenizer.index_word[len(target_tokenizer.word_index) + 2] = end_token
target_tokenizer.index_word[len(target_tokenizer.word_index) + 3] = unknown_token

# Get vocabulary sizes
source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Maximum sequence lengths
max_source_length = max(len(seq.split()) for seq in train_df['source'])
max_target_length = max(len(seq.split()) for seq in train_df['target'])

# Create input sequences and target sequences for training
encoder_input_data = source_tokenizer.texts_to_sequences(train_df['source'])
decoder_input_data = target_tokenizer.texts_to_sequences(train_df['target'])
decoder_target_data = [seq[1:] for seq in decoder_input_data]

# Pad sequences for uniform length
encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_source_length, padding='post')
decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_target_length, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_target_length, padding='post')

# Split the test data
test_encoder_input_data = pad_sequences(source_tokenizer.texts_to_sequences(test_df['source']), maxlen=max_source_length, padding='post')
test_decoder_input_data = pad_sequences(target_tokenizer.texts_to_sequences(test_df['target']), maxlen=max_target_length, padding='post')
test_decoder_target_data = pad_sequences([seq[1:] for seq in target_tokenizer.texts_to_sequences(test_df['target'])], maxlen=max_target_length, padding='post')

# Embedding layers
embedding_layer_source = Embedding(source_vocab_size, embedding_dim)
embedding_layer_target = Embedding(target_vocab_size, embedding_dim)

# Encoder
encoder_inputs = tf.keras.Input(shape=(None,))
encoder_embedding = embedding_layer_source(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.Input(shape=(None,))
decoder_embedding = embedding_layer_target(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

def translate_sentence(input_sentence):
    # Encode the input sentence
    input_seq = source_tokenizer.texts_to_sequences([input_sentence])
    input_seq = pad_sequences(input_seq, maxlen=max_source_length, padding='post')
    encoder_output, state_h, state_c = encoder_lstm(embedding_layer_source(input_seq))
    encoder_states = [state_h, state_c]

    # Start the decoding process
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index[start_token]

    stop_condition = False
    decoded_sentence = ''
    prev_sampled_token_index = None  # Keep track of the previous sampled token

    while not stop_condition:
        output_tokens, h, c = decoder_lstm(embedding_layer_target(target_seq), initial_state=encoder_states)
        output_tokens = decoder_dense(output_tokens)
        sampled_token_index = tf.random.categorical(tf.math.log(output_tokens[0]), 1).numpy()[0, 0]

        if sampled_token_index == 0 or sampled_token_index >= target_vocab_size:
            # Handle unknown token or padding
            sampled_token_index = target_tokenizer.word_index[unknown_token]

        sampled_char = target_tokenizer.index_word[sampled_token_index]

        if sampled_char != end_token:
            decoded_sentence += ' ' + sampled_char

        if sampled_char == end_token or len(decoded_sentence.split()) > max_target_length or sampled_token_index == prev_sampled_token_index:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        encoder_states = [h, c]
        prev_sampled_token_index = sampled_token_index  # Update the previous sampled token

    return decoded_sentence.strip()


if __name__ == '__main__':
    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer=RMSprop(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Train the model
    epochs = 25
    batch_size = 64

    # Lists to store metrics
    epoch_list = []
    train_test_split_list = []
    size_of_dataset_list = []
    training_accuracy_list = []
    test_accuracy_list = []
    training_rmse_list = []
    test_rmse_list = []

    for epoch in range(epochs):
        for batch_start in range(0, len(encoder_input_data), batch_size):
            batch_end = batch_start + batch_size
            encoder_batch = encoder_input_data[batch_start:batch_end]
            decoder_batch = decoder_input_data[batch_start:batch_end]
            target_batch = decoder_target_data[batch_start:batch_end]

            loss, accuracy = model.train_on_batch([encoder_batch, decoder_batch], target_batch)

        # Calculate training accuracy for the epoch
        training_accuracy = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data, verbose=0)[1]

        # Calculate test accuracy for the epoch
        test_accuracy = model.evaluate([test_encoder_input_data, test_decoder_input_data], test_decoder_target_data, verbose=0)[1]

        # Calculate training RMSE for the epoch
        train_rmse = np.sqrt(mean_squared_error(decoder_target_data.flatten(), model.predict([encoder_input_data, decoder_input_data]).argmax(axis=-1).flatten()))

        # Calculate test RMSE for the epoch
        test_rmse = np.sqrt(mean_squared_error(test_decoder_target_data.flatten(), model.predict([test_encoder_input_data, test_decoder_input_data]).argmax(axis=-1).flatten()))

        # Append metrics to lists
        epoch_list.append(epoch + 1)
        train_test_split_list.append('90:10')
        size_of_dataset_list.append(len(subset_df))
        training_accuracy_list.append(training_accuracy)
        test_accuracy_list.append(test_accuracy)
        training_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)

        print(f'Epoch {epoch + 1}/{epochs}, Training Accuracy: {training_accuracy}, Test Accuracy: {test_accuracy}, Training RMSE: {train_rmse}, Test RMSE: {test_rmse}')
        
    model.save('seq2seq_model.h5')
    
    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'Epoch': epoch_list,
        'Train/Test Split': train_test_split_list,
        'Size of Dataset': size_of_dataset_list,
        'Training Accuracy': training_accuracy_list,
        'Test Accuracy': test_accuracy_list,
        'Training RMSE': training_rmse_list,
        'Test RMSE': test_rmse_list
    })

    # Save the metrics DataFrame to a CSV
    metrics_df.to_csv('metrics.csv', index=False)
    # Test the model with some input sentences
    for i in range(5):
        input_sentence = test_df['source'].iloc[i]
        expected_output = test_df['target'].iloc[i]

        translated_sentence = translate_sentence(input_sentence)
        print(f'Input: {input_sentence}')
        print(f'Expected Output: {expected_output}')
        print(f'Translated Output: {translated_sentence}\n')
    