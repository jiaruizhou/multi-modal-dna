conda create -n dna python=3.11
conda install pytorch==2.5.0 torchvision==0.20.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers timm tensorboard scikit-learn biopython matplotlib