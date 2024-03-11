from datetime import datetime, timedelta
import bcrypt
import torch
import re
import unicodedata
from io import open
import itertools

from utils.vocab_generator import Voc


def check_hash(content: str, hashed_content: str) -> bool:
    return bcrypt.checkpw(bytes(content, 'utf-8'), bytes(hashed_content, 'utf-8'))


def hashing(content: str):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(bytes(content, 'utf-8'), salt)
    return hashed.decode("utf-8")


def current_date() -> str:
    dt = datetime.now().strftime('%m%d%Y')
    return dt


def current_date_time() -> str:
    dt = datetime.now().strftime('%m%d%Y_%H%M%S')
    return dt


def cookie_expiration_set():
    expire_date = datetime.now()
    expire_date = expire_date + timedelta(days=1)
    return expire_date


def strict_ip_filer(ip_data: dict, fields: set):
    if all(name in ip_data for name in fields):
        return ip_data
    else:
        raise ValueError("Keys are not matching")


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def unicodeTo_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalize_string(text):
    text = unicodeTo_ascii(text.lower().strip())
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("(\\W)", " ", text)
    text = re.sub('\S*\d\S*\s*', '', text)
    # text =  "<sos> " +  text + " <eos>"
    return text


# Read query/response pairs and return a voc object
def read_vocabs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the max_length threshold
def filter_pair(p, max_length):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) <= max_length and len(p[1].split(' ')) <= max_length


# Filter pairs using filterPair condition
def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus_name, datafile, max_length):
    print("Start preparing training data ...")
    voc, pairs = read_vocabs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs, max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Remove words with count lesser than the min_count
def trim_rare_words(voc, pairs, min_count):
    # Trim words used under the min_count from the voc
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


# Get index of words from a sentence
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Pad the tensors having lesser entries with 0s
def zeroPadding(l, fill_value=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fill_value))


# Create mask for the indexes
def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


######################################

def input_var(l, word2index):
    indexes_batch = [indexesFromSentence_v(word2index, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def output_var(l, word2index):
    indexes_batch = [indexesFromSentence_v(word2index, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    mask = binaryMatrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len


def split_inp_out(pair_batch: list, word2index: dict):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, word2index)
    output, mask, max_target_len = output_var(output_batch, word2index)
    return inp, lengths, output, mask, max_target_len


def indexesFromSentence_v(word2index, sentence):
    return [word2index[word] for word in sentence.split(' ')] + [EOS_token]


def maskNLLLoss(inp, target, mask, args):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(args.device)
    return loss, n_total.item()
