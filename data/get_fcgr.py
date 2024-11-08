import os
import torch
from torch.utils.data import Dataset
from Bio import SeqIO  # 请确保安装了Biopython库
from FCGR import fcgr
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('FCGR GENERATION', add_help=False)

    parser.add_argument('--data_path', default="./pretraining_data", type=str)
    parser.add_argument('--dataset', default="pretrain", type=str)
    # parser.add_argument('--data', default="", type=str)
    parser.add_argument('--kmer', default=3, type=int)  # input size 2**k x 2**k

    return parser


def load_fcgr(fcgr_path, records):

    samples = []
    seqs = [str(record.seq) for record in records]
    print("start fcgr")
    samples.extend([torch.tensor(fcgr(sequence, sequence, args.kmer)).unsqueeze(0) for sequence in seqs])
    torch.save(samples, fcgr_path)
    return samples


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    if args.dataset != "pretrain":
        for phase in ["train", "test"]:
            fcgr_features = []
            file = os.path.join(args.data_path, 'train.fa' if phase == "train" else 'test.fa')  # ./similar

            fcgr_path = file.split(".fa")[0] + '_{}mer.npy'.format(args.kmer)
            if os.path.exists(fcgr_path):
                print("= = = = the fcgr data existed = = = = =")
            else:
                records = list(SeqIO.parse(file, "fasta"))
                fcgr_features = load_fcgr(fcgr_path, records)
            print(args.dataset)
            print("phase is ok", phase)
            torch.save(fcgr_features, fcgr_path)

            print("Number of Samples:", len(fcgr_features))
    else:
        classes = [d.split("_db.fa")[0] for d in os.listdir(args.data_path)]  # ./pretraining_data

        if os.path.exists("./pretrain_fcgr/samples_{}mer.npy".format(args.kmer)):
            print("sample for k{} extisted".format(args.kmer))
        else:
            fcgr_features = []
            for class_name in classes:
                class_dir = os.path.join(args.data_path, class_name + "_db.fa")
                sequences = [str(record.seq) for record in SeqIO.parse(class_dir, "fasta")]
                fcgr_features.extend([(torch.tensor(fcgr(sequence, sequence, args.kmer)).unsqueeze(0)) for sequence in sequences])
                print("class db is ok", class_name)
            torch.save(fcgr_features, "./pretrain_fcgr/samples_{}mer.npy".format(args.kmer))
        print(args.dataset)
        print("Number of Samples:", len(fcgr_features))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    main(args)
