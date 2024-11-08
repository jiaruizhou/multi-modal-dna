import numpy as np
from random import randint
from itertools import product
from typing import List, OrderedDict, Optional, Tuple, Union
from logging import info
import numpy as np
from random import sample
import argparse
import warnings
import collections
import torch
#  TODO env:bertax2
# from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, BertConfig, BertModel

warnings.simplefilter(action='ignore', category=FutureWarning)

ALPHABET = 'ACGT'
CLASS_LABELS = OrderedDict([
    ('superkingdom', ['Archaea', 'Bacteria', 'Eukaryota', 'Viruses', 'unknown']),
    ('phylum', [
        'Actinobacteria', 'Apicomplexa', 'Aquificae', 'Arthropoda', 'Artverviricota', 'Ascomycota', 'Bacillariophyta', 'Bacteroidetes', 'Basidiomycota', 'Candidatus Thermoplasmatota', 'Chlamydiae', 'Chlorobi', 'Chloroflexi', 'Chlorophyta',
        'Chordata', 'Crenarchaeota', 'Cyanobacteria', 'Deinococcus-Thermus', 'Euglenozoa', 'Euryarchaeota', 'Evosea', 'Firmicutes', 'Fusobacteria', 'Gemmatimonadetes', 'Kitrinoviricota', 'Lentisphaerae', 'Mollusca', 'Negarnaviricota', 'Nematoda',
        'Nitrospirae', 'Peploviricota', 'Pisuviricota', 'Planctomycetes', 'Platyhelminthes', 'Proteobacteria', 'Rhodophyta', 'Spirochaetes', 'Streptophyta', 'Tenericutes', 'Thaumarchaeota', 'Thermotogae', 'Uroviricota', 'Verrucomicrobia', 'unknown'
    ]),
    ('genus', [
        'Acidilobus', 'Acidithiobacillus', 'Actinomyces', 'Actinopolyspora', 'Acyrthosiphon', 'Aeromonas', 'Akkermansia', 'Anas', 'Apis', 'Aquila', 'Archaeoglobus', 'Asparagus', 'Aspergillus', 'Astyanax', 'Aythya', 'Bdellovibrio', 'Beta', 'Betta',
        'Bifidobacterium', 'Botrytis', 'Brachyspira', 'Bradymonas', 'Brassica', 'Caenorhabditis', 'Calypte', 'Candidatus Kuenenia', 'Candidatus Nitrosocaldus', 'Candidatus Promineofilum', 'Carassius', 'Cercospora', 'Chanos', 'Chlamydia', 'Chrysemys',
        'Ciona', 'Citrus', 'Clupea', 'Coffea', 'Colletotrichum', 'Cottoperca', 'Crassostrea', 'Cryptococcus', 'Cucumis', 'Cucurbita', 'Cyanidioschyzon', 'Cynara', 'Cynoglossus', 'Daucus', 'Deinococcus', 'Denticeps', 'Desulfovibrio', 'Dictyostelium',
        'Drosophila', 'Echeneis', 'Egibacter', 'Egicoccus', 'Elaeis', 'Equus', 'Erpetoichthys', 'Esox', 'Euzebya', 'Fervidicoccus', 'Frankia', 'Fusarium', 'Gadus', 'Gallus', 'Gemmata', 'Gopherus', 'Gossypium', 'Gouania', 'Helianthus', 'Ictalurus',
        'Ktedonosporobacter', 'Legionella', 'Leishmania', 'Lepisosteus', 'Leptospira', 'Limnochorda', 'Malassezia', 'Manihot', 'Mariprofundus', 'Methanobacterium', 'Methanobrevibacter', 'Methanocaldococcus', 'Methanocella', 'Methanopyrus',
        'Methanosarcina', 'Microcaecilia', 'Modestobacter', 'Monodelphis', 'Mus', 'Musa', 'Myripristis', 'Neisseria', 'Nitrosopumilus', 'Nitrososphaera', 'Nitrospira', 'Nymphaea', 'Octopus', 'Olea', 'Oncorhynchus', 'Ooceraea', 'Ornithorhynchus',
        'Oryctolagus', 'Oryzias', 'Ostreococcus', 'Papaver', 'Perca', 'Phaeodactylum', 'Phyllostomus', 'Physcomitrium', 'Plasmodium', 'Podarcis', 'Pomacea', 'Populus', 'Prosthecochloris', 'Pseudomonas', 'Punica', 'Pyricularia', 'Pyrobaculum',
        'Quercus', 'Rhinatrema', 'Rhopalosiphum', 'Roseiflexus', 'Rubrobacter', 'Rudivirus', 'Salarias', 'Salinisphaera', 'Sarcophilus', 'Schistosoma', 'Scleropages', 'Sedimentisphaera', 'Sesamum', 'Solanum', 'Sparus', 'Sphaeramia', 'Spodoptera',
        'Sporisorium', 'Stanieria', 'Streptomyces', 'Strigops', 'Synechococcus', 'Takifugu', 'Thalassiosira', 'Theileria', 'Thermococcus', 'Thermogutta', 'Thermus', 'Tribolium', 'Trichoplusia', 'Ustilago', 'Vibrio', 'Vitis', 'Xenopus', 'Xiphophorus',
        'Zymoseptoria', 'unknown'
    ])
])

Record = collections.namedtuple('Record', ['id', 'seq'])


def parse_fasta(fasta) -> List[Record]:
    records = []
    id_ = None
    seq = ''
    allowed_characters = set("ACTGYRWSKMDVHBXN")
    with open(fasta) as f:
        for line in f:
            if line.startswith('>'):
                if (id_ is not None):
                    records.append(Record(id_, seq))
                id_ = line[1:].strip()
                seq = ''
            else:
                line_no_whitespace = line.strip()
                assert set(line_no_whitespace.upper()).issubset(allowed_characters), \
                    f"{fasta} is not a fasta-file: contains not allowed characters " \
                    f"{set(line_no_whitespace.upper()).difference(allowed_characters)} in line\n{line_no_whitespace}"
                seq += line_no_whitespace

        assert id_ is not None, f"{fasta} is not a fasta file: no header detected"
        records.append(Record(id_, seq))
    info(f'read in {len(records)} sequences')
    return records

def seq2kmers(seq, k=3, stride=3, pad=True, to_upper=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if (len(seq) < k):
        return [seq.ljust(k, 'N')] if pad else []
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i:i + k]
        if to_upper:
            kmers.append(kmer.upper())
        else:
            kmers.append(kmer)
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i + k:].ljust(k, 'N'))
    return kmers


def seq2tokens(seq, token_dict, seq_length=250, max_length=None, k=3, stride=3, window=True, seq_len_like=None):
    """transforms raw sequence into list of tokens to be used for
    fine-tuning BERT"""
    if (max_length is None):
        max_length = seq_length
    if (seq_len_like is not None):
        seq_length = min(max_length, np.random.choice(seq_len_like))
        # open('seq_lens.txt', 'a').write(str(seq_length) + ', ')
    seq = seq2kmers(seq, k=k, stride=stride, pad=True)
    if (window):
        start = randint(0, max(len(seq) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
    indices = [token_dict['[CLS]']] + [token_dict[word] if word in token_dict else token_dict['[UNK]'] for word in seq[start:end]]
    if (len(indices) < max_length):
        indices += [token_dict['']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]


def seq_frames(seq: str, frame_len: int, running_window=False, stride=1):
    """returns all windows of seq with a maximum length of `frame_len` and specified stride
    if `running_window` else `frame_len` -- alongside the chunks' positions"""
    iterator = (range(0, len(seq) - frame_len + 1, stride) if running_window else range(0, len(seq), frame_len))
    return [seq[i:i + frame_len] for i in iterator], [(i, i + frame_len) for i in iterator]


def get_token_dict(alph=ALPHABET, k=3) -> dict:
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = {'': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

    for word in [''.join(_) for _ in product(alph, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict

def process_bert_tokens_batch(batch_x):
    """when `seq2tokens` is used as `custom_encode_sequence`, batches
    are generated as [[input1, input2], [input1, input2], ...]. In
    order to train, they have to be transformed to [input1s,
    input2s] with this function"""
    return [np.array([_[0] for _ in batch_x]), np.array([_[1] for _ in batch_x])]


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
    return torch.tensor(np.array(x[0])),torch.tensor(np.array(x[1]))# ( , 1, 502)