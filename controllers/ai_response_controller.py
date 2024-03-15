from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from utils.helpers import indexesFromSentence, normalize_string, trim_rare_words, loadPrepareData
from datetime import datetime
from utils.chat_net import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
import os

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# import string
# import joblib
# from pandas import DataFrame, concat

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def evaluate(searcher, voc, sentence, device):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate args.device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(searcher, voc, input_sentence, device):
    while 1:
        try:
            # Get input sentence
            # input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence, device)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
            return ' '.join(output_words)

        except KeyError:
            print("Error: Encountered unknown word.")


def response_main(input_sentence: str):
    hidden_size = 500
    attn_model = 'dot'  # (dot/general/concat)
    min_count = 3
    max_length = 20
    encoder_n_layers = 2
    decoder_n_layers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_length = 20

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    save_dir = os.path.join("saved_models", "checkpoints")
    # Configure models
    model_name = 'model_vision'
    dropout = 0.1

    # Set checkpoint to load from; set to None if starting from scratch

    # Load model if a loadFilename is provided
    # If loading on same machine the model was trained on
    checkpoint = torch.load('data_models/vision_checkpoint.tar')
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.args.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = checkpoint['voc']
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers,
                                  dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate args.device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.eval()
    decoder.eval()


    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, device, max_length)

    # Begin chatting (uncomment and run the following line to begin)
    output_ = evaluateInput(searcher, voc, input_sentence, device)
    return output_



