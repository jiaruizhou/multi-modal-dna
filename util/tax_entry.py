# import sqlite3
# from ete3 import NCBITaxa
import os
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from util.FCGR import fcgr
import numpy as np
from typing import List, OrderedDict, Optional

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
        'Zymoseptoria'
    ])
])

supk_dict = {'Archaea': 0, 'Bacteria': 1, 'Eukaryota': 2, 'Viruses': 3, 'unknown': 4}
phyl_dict = {
    'Aquificae': 0,
    'Tenericutes': 1,
    'Actinobacteria': 2,
    'Chlorophyta': 3,
    'Deinococcus-Thermus': 4,
    'Nematoda': 5,
    'Chordata': 6,
    'Basidiomycota': 7,
    'Crenarchaeota': 8,
    'Proteobacteria': 9,
    'Verrucomicrobia': 10,
    'Ascomycota': 11,
    'Bacillariophyta': 12,
    'Negarnaviricota': 13,
    'Rhodophyta': 14,
    'Gemmatimonadetes': 15,
    'Peploviricota': 16,
    'Uroviricota': 17,
    'Kitrinoviricota': 18,
    'Nitrospirae': 19,
    'Lentisphaerae': 20,
    'Platyhelminthes': 21,
    'Arthropoda': 22,
    'Streptophyta': 23,
    'Thermotogae': 24,
    'Pisuviricota': 25,
    'Euryarchaeota': 26,
    'Mollusca': 27,
    'Euglenozoa': 28,
    'Planctomycetes': 29,
    'Evosea': 30,
    'Artverviricota': 31,
    'Chlorobi': 32,
    'Firmicutes': 33,
    'Chloroflexi': 34,
    'Candidatus Thermoplasmatota': 35,
    'Chlamydiae': 36,
    'Cyanobacteria': 37,
    'Bacteroidetes': 38,
    'Thaumarchaeota': 39,
    'Apicomplexa': 40,
    'Fusobacteria': 41,
    'unknown': 42,
    'Spirochaetes': 43
}


genus_dict = {
    'Microcaecilia': 0,
    'Roseiflexus': 1,
    'Schistosoma': 2,
    'Euzebya': 3,
    'Colletotrichum': 4,
    'Gallus': 5,
    'Strigops': 6,
    'Methanosarcina': 7,
    'Nitrospira': 8,
    'Botrytis': 9,
    'Asparagus': 10,
    'Sparus': 11,
    'Fervidicoccus': 12,
    'Dictyostelium': 13,
    'Bdellovibrio': 14,
    'Oryctolagus': 15,
    'Takifugu': 16,
    'Punica': 17,
    'Cynara': 18,
    'Aspergillus': 19,
    'Olea': 20,
    'Rhopalosiphum': 21,
    'Esox': 22,
    'Ostreococcus': 23,
    'Brassica': 24,
    'Echeneis': 25,
    'Aythya': 26,
    'Egicoccus': 27,
    'Astyanax': 28,
    'Nitrosopumilus': 29,
    'Pomacea': 30,
    'Daucus': 31,
    'Pyricularia': 32,
    'Vitis': 33,
    'Trichoplusia': 34,
    'Elaeis': 35,
    'Plasmodium': 36,
    'Mus': 37,
    'Actinopolyspora': 38,
    'Ciona': 39,
    'Theileria': 40,
    'Lepisosteus': 41,
    'Methanopyrus': 42,
    'Helianthus': 43,
    'Cyanidioschyzon': 44,
    'Sarcophilus': 45,
    'Legionella': 46,
    'Gadus': 47,
    'unknown': 48,
    'Archaeoglobus': 49,
    'Drosophila': 50,
    'Rubrobacter': 51,
    'Fusarium': 52,
    'Leptospira': 53,
    'Spodoptera': 54,
    'Chanos': 55,
    'Limnochorda': 56,
    'Methanobacterium': 57,
    'Candidatus Promineofilum': 58,
    'Gopherus': 59,
    'Stanieria': 60,
    'Solanum': 61,
    'Calypte': 62,
    'Thermus': 63,
    'Beta': 64,
    'Sedimentisphaera': 65,
    'Rudivirus': 66,
    'Phyllostomus': 67,
    'Quercus': 68,
    'Neisseria': 69,
    'Akkermansia': 70,
    'Sesamum': 71,
    'Cucumis': 72,
    'Ictalurus': 73,
    'Ktedonosporobacter': 74,
    'Anas': 75,
    'Citrus': 76,
    'Cynoglossus': 77,
    'Aquila': 78,
    'Oryzias': 79,
    'Brachyspira': 80,
    'Papaver': 81,
    'Apis': 82,
    'Methanocella': 83,
    'Leishmania': 84,
    'Cercospora': 85,
    'Egibacter': 86,
    'Cryptococcus': 87,
    'Equus': 88,
    'Salinisphaera': 89,
    'Streptomyces': 90,
    'Gemmata': 91,
    'Octopus': 92,
    'Sporisorium': 93,
    'Pseudomonas': 94,
    'Deinococcus': 95,
    'Thermococcus': 96,
    'Gossypium': 97,
    'Betta': 98,
    'Aeromonas': 99,
    'Thermogutta': 100,
    'Frankia': 101,
    'Thalassiosira': 102,
    'Crassostrea': 103,
    'Acyrthosiphon': 104,
    'Denticeps': 105,
    'Chlamydia': 106,
    'Cottoperca': 107,
    'Acidithiobacillus': 108,
    'Ornithorhynchus': 109,
    'Cucurbita': 110,
    'Podarcis': 111,
    'Malassezia': 112,
    'Xiphophorus': 113,
    'Perca': 114,
    'Actinomyces': 115,
    'Modestobacter': 116,
    'Synechococcus': 117,
    'Musa': 118,
    'Oncorhynchus': 119,
    'Methanobrevibacter': 120,
    'Pyrobaculum': 121,
    'Vibrio': 122,
    'Tribolium': 123,
    'Desulfovibrio': 124,
    'Scleropages': 125,
    'Ooceraea': 126,
    'Sphaeramia': 127,
    'Nymphaea': 128,
    'Zymoseptoria': 129,
    'Acidilobus': 130,
    'Candidatus Kuenenia': 131,
    'Chrysemys': 132,
    'Phaeodactylum': 133,
    'Salarias': 134,
    'Nitrososphaera': 135,
    'Coffea': 136,
    'Clupea': 137,
    'Bifidobacterium': 138,
    'Ustilago': 139,
    'Physcomitrium': 140,
    'Populus': 141,
    'Gouania': 142,
    'Carassius': 143,
    'Rhinatrema': 144,
    'Mariprofundus': 145,
    'Monodelphis': 146,
    'Candidatus Nitrosocaldus': 147,
    'Prosthecochloris': 148,
    'Xenopus': 149,
    'Erpetoichthys': 150,
    'Methanocaldococcus': 151,
    'Bradymonas': 152,
    'Caenorhabditis': 153,
    'Manihot': 154,
    'Myripristis': 155
}

new_supk_dict = {v: k for k, v in supk_dict.items()}
new_phyl_dict = {v: k for k, v in phyl_dict.items()}
new_genus_dict = {v: k for k, v in genus_dict.items()}


class TaxidLineage:

    def __init__(self):
        # self.ncbi = NCBITaxa()
        self.cache = {}
        # self.ncbi.db = sqlite3.connect(self.ncbi.dbfile, check_same_thread=False)

    def populate(self, taxids, ranks=['superkingdom', 'kingdom', 'phylum', 'family']):
        for taxid in taxids:
            d = self.ncbi.get_rank(self.ncbi.get_lineage(taxid))
            self.cache[taxid] = {r: self._get_d_rank(d, r) for r in ranks}

    def _get_d_rank(self, d, rank):
        if (rank not in d.values()):
            return (None, 'unknown')
        taxid = [k for k, v in d.items() if v == rank]
        name = self.ncbi.translate_to_names(taxid)[0]
        return (taxid[0], name if isinstance(name, str) else 'unknown')

    def get_ranks(self, taxid, ranks=['superkingdom', 'kingdom', 'phylum', 'family']):
        if taxid in self.cache:
            return self.cache[taxid]
        d = self.ncbi.get_rank(self.ncbi.get_lineage(taxid))
        return {r: self._get_d_rank(d, r) for r in ranks}


def annotate_predictions(preds):
    """annotates list of prediction arrays with provided or preset labels"""
    return {int(ind): [new_supk_dict[int(pred0)], new_phyl_dict[int(pred1)], new_genus_dict[int(pred2)]] for (ind, pred0, pred1, pred2) in preds}
