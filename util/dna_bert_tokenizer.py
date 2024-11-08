import warnings
#  TODO env:bertax2
# from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, BertConfig, BertModel

warnings.simplefilter(action='ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
from ai4.multi_modal_dna.data.bertax_data import (seq2tokens, get_token_dict, process_bert_tokens_batch, seq_frames, CLASS_LABELS)

from logging import info
import numpy as np
from random import sample

MAX_SIZE = 1500
FIELD_LEN_THR = 50  # prevent overlong IDs

def check_max_len_arg(value):
    value = int(value)
    if (value < 1 or value > MAX_SIZE):
        raise argparse.ArgumentTypeError(f'value has to be between 1 and {MAX_SIZE}')
    return value


def check_ranks(value):
    if (value not in CLASS_LABELS):
        raise argparse.ArgumentTypeError(f'output ranks have to be a combination of {set(CLASS_LABELS)}, '
                                         'other ranks are not predicted')
    return value

def convert_tokens_to_ids(records,args):
    # 获取倒数第6层的输出
    custom_window_size = None
    running_window = None
    running_window_stride = 1
    sequence_split = 'window'
    maximum_sequence_chunks = 500
    max_len=args.seq_len
    max_seq_len = MAX_SIZE if custom_window_size is None else custom_window_size
    token_dict = get_token_dict()
    x = [[], []]

    for record in records:
        no_chunks = (not running_window and (len(record.seq) <= max_seq_len or sequence_split == 'window'))
        if (no_chunks):
            inputs = [seq2tokens(record.seq, token_dict, np.ceil(max_seq_len / 3).astype(int), max_length=max_len)]
        else:
            chunks, positions = seq_frames(record.seq, max_seq_len, running_window, running_window_stride)
            if (maximum_sequence_chunks > 0 and len(chunks) > maximum_sequence_chunks):
                info(f'sampling {maximum_sequence_chunks} from {len(chunks)} chunks')
                chunks, positions = list(zip(*sample(list(zip(chunks, positions)), maximum_sequence_chunks)))
            inputs = [seq2tokens(chunk, token_dict, np.ceil(max_seq_len / 3).astype(int), max_length=max_len) for chunk in chunks]
        info(f'converted sequence {record.id} (length {len(record.seq)}) into {len(inputs)} chunks')
        x0, x1 = process_bert_tokens_batch(inputs)
        x[0].append(x0)
        x[1].append(x1)
        
    return x