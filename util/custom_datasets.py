import os
import PIL
import torch
from torch.utils.data import Dataset
from Bio import SeqIO  # 请确保安装了Biopython库
from util.FCGR import fcgr
from util.tax_entry import  supk_dict, phyl_dict, genus_dict, new_supk_dict, new_phyl_dict, new_genus_dict
from util.tax_entry import TaxidLineage
import numpy as np
from util.bertax_data import convert_tokens_to_ids,parse_fasta


class Visual_Text_Dataset(Dataset):

    def __init__(self, args, files, kmer, phase="train", include_textual_data=True,transform=None):
        self.kmer = kmer
        self.fcgr_features = [] # for the fcgr
        self.samples = []
        # load the textual feature from bertax
        # print("*"*10,"Load the textual feature","*"*10)

        # # self.textual_features=torch.load("/aiarena/group/mmirgroup/zhoujr/ai4/bertax_nf/outputs/{}/{}.pt".format(args.data,phase))  # used for textual pooling features

        self.file = os.path.join(files, 'train.fa' if phase == "train" else 'test.fa')
        supk_path = self.file.split(".fa")[0] + '_cls_{}.npy'.format("superkingdom")
        phyl_path = self.file.split(".fa")[0] + '_cls_{}.npy'.format("phylum")
        genus_path = self.file.split(".fa")[0] + '_cls_{}.npy'.format("genus")
        #  #  get the fcgr features
        fcgr_path = self.file.split(".fa")[0] + '_{}mer.pt'.format(self.kmer)

        token_ids_path = self.file.split(".fa")[0] + '_token_ids.pt'
        token_type_ids_path = self.file.split(".fa")[0] + '_token_type_ids.pt'

        self.fcgr_features = self._get_fcgr(fcgr_path)
        # # load the class list if exists
        print("*"*10,"Load the class labels","*"*10)
        genus_names = torch.load(genus_path)
        genus_set = set(genus_dict.keys())
        genus_names = [genus_dict['unknown'] if element not in genus_set else genus_dict[element] for element in genus_names]

        supk_names = torch.load(supk_path)
        supk_set = set(supk_dict.keys())
        supk_names = [supk_dict['unknown'] if element not in supk_set else supk_dict[element] for element in supk_names]
        phyl_names = torch.load(phyl_path)
        phyl_set = set(phyl_dict.keys())
        phyl_names = [phyl_dict['unknown'] if element not in phyl_set else phyl_dict[element] for element in phyl_names]
        
        print("*"*5,"Dataset includes {} supk {} phyl {} genus {}".format(self.file,len(set(supk_names)),len(set(phyl_names)),len(set(genus_names))),"*"*5)

        if not include_textual_data:
            
            self.textual_features=torch.load("/aiarena/group/mmirgroup/zhoujr/ai4/bertax_nf/outputs/{}/{}.pt".format(args.data,phase))  # used for textual pooling features
            self.samples.extend([((fcgr,text_feature), \
                (cls_name1,cls_name2,cls_name3)) \
                    for _,(fcgr, text_feature,cls_name1,cls_name2,cls_name3) in \
                        enumerate(zip(self.fcgr_features, self.textual_features,torch.tensor(supk_names)\
                            ,torch.tensor(phyl_names),torch.tensor(genus_names)))])
        
        elif include_textual_data:
            print("*"*10,"Load the textual feature","*"*10)
            self.token_ids,self.token_type_ids=self.__seq_to_ids(args,token_ids_path,token_type_ids_path)

            self.samples.extend([((fcgr,(token_ids,token_type_ids)), \
                (cls_name1,cls_name2,cls_name3)) \
                for _,(fcgr, token_ids,token_type_ids,cls_name1,cls_name2,cls_name3) in \
                enumerate(zip(self.fcgr_features, self.token_ids,self.token_type_ids,torch.tensor(supk_names)\
                    ,torch.tensor(phyl_names),torch.tensor(genus_names)))])
        
        print("Length of The Dataset:", len(self.samples))

    def __seq_to_ids(self,args,token_ids_path,token_type_ids_path):
        
        if os.path.exists(token_ids_path) and os.path.exists(token_type_ids_path):
            token_ids=torch.load(token_ids_path)
            token_type_ids=torch.load(token_type_ids_path)
        else:
            records = parse_fasta(self.file)
            token_ids,token_type_ids=convert_tokens_to_ids(records,args)
            torch.save(token_ids,token_ids_path)
            torch.save(token_type_ids,token_type_ids_path)
        return token_ids,token_type_ids 


    def _get_fcgr(self, fcgr_path):
        samples = []

        if os.path.exists(fcgr_path):
            print("*"*10,"Load the fcgr data","*"*10)
            samples = torch.load(fcgr_path)
        else:
            records = list(SeqIO.parse(self.file, "fasta"))            
            self.seqs = [str(record.seq) for record in records]
            print("*"*10,"start the process for DNA sequence to fcgr","*"*10)
            samples.extend([torch.tensor(fcgr(sequence, sequence, self.kmer)).unsqueeze(0) for sequence in self.seqs])
            torch.save(samples, fcgr_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input, input_id = self.samples[idx]
        return input, input_id
