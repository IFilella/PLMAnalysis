import torch
import esm
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_tensor_shape(tensor, device):
    if device == 'cuda':
        shape = tensor.cpu().detach().numpy().shape
    elif device == 'cpu':
        shape = tensor.numpy().shape
    return shape


def read_fasta(fasta, unalign=False, delimiter='None'):
    f = open(fasta, 'r')
    seqs = []
    titles = []
    for line in f:
        if line[0] == '>':
            titles.append(line.replace('\n', '').replace('>', ''))
        else:
            if unalign:
                seqs.append(line.replace('\n', '').replace('-', ''))
            else:
                seqs.append(line.replace('\n', ''))
    if delimiter is not None:
        labels = [title.split(delimiter)[0] for title in titles]
        return seqs, titles, labels
    else:
        return seqs, titles

def get_GPU_memory():
    current_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    max_reserved_memory = torch.cuda.max_memory_reserved()
    print('.........GPU Memory.........')
    print(f"Current GPU memory usage: {current_memory / 1024 / 1024:.2f} MB")
    print(f"Reserved GPU memory: {reserved_memory / 1024 / 1024:.2f} MB")
    print(f"Max Reserved GPU memory: {max_reserved_memory / 1024 / 1024:.2f} MB")
    free, total = torch.cuda.mem_get_info()
    print(f'Free {free / 1024 / 1024:.3f} MB ')
    print(f'Total {total / 1024 / 1024:.3f} MB ')
    print('...........................')


# Paramaters
batch_size = 10
verbose = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
if verbose:
    print(device)
    get_GPU_memory()

# Load Sequences
msa = '../data/T6SS_shuffle.fasta'
#msa = '../data/human_kinome_noPLK5.aln'
seqs, titles, labels = read_fasta(msa, unalign=True, delimiter='_')
data = [(titles[i], seqs[i]) for i in range(len(seqs))]
"""
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
#model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
#model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
if verbose:
    print('After loading the model')
    get_GPU_memory()

seqs_embeddings = []
for i in range(0, len(data), batch_size):
    print('%d / %d' % (i, len(data)))
    batch = data[i:i + batch_size]
    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    if verbose:
        print('After tokenizing')
        get_GPU_memory()
        #print(batch_labels)
        #print(batch_strs)
        #print(batch_tokens)
        shape = get_tensor_shape(batch_tokens, device)
        print(shape)

    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    if verbose:
        print('After getting the embedding')
        get_GPU_memory()
        #print(token_representations)
        shape = get_tensor_shape(token_representations, device)
        print(shape)

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        seq_token_representations = token_representations[i, 1 : tokens_len - 1].mean(0)
        if device == 'cuda':
            seq_token_representations = seq_token_representations.to('cpu')
        sequence_representations.append(seq_token_representations)
    if verbose:
        shape = get_tensor_shape(sequence_representations[0], device)
        print(shape)
    for sequence_representation in sequence_representations:
        seqs_embeddings.append(sequence_representation)

with open('../results/T6SS_CPU_test.pickle', 'wb') as handle:
    pickle.dump(seqs_embeddings, handle,
                protocol=pickle.HIGHEST_PROTOCOL)
exit()
"""

with open("../results/T6SS_GPU_test.pickle", "rb") as input_file:
    seqs_embeddings2 = pickle.load(input_file)
seqs_embeddings2 = np.asarray(seqs_embeddings2)
print(len(seqs_embeddings2[0]))

from sklearn.manifold import TSNE
perps = [10,15,20,30,40]
metrics = ['l2', 'braycurtis', 'correlation', 'l1', 'manhattan', 'euclidean', 'cityblock' , 'minkowski','sqeuclidean', 'cosine', 'minkowski', 'nan_euclidean', 'canberra']
perps = [15]
metrics = ['canberra']
for perp in perps:
    for m in metrics:
        tsne=TSNE(n_components=2, verbose=1, learning_rate='auto', init='pca',
                  perplexity=perp, n_iter=1500, early_exaggeration=12, metric=m)
        tsne_results=tsne.fit_transform(seqs_embeddings2)
        df=pd.DataFrame(dict(xaxis=tsne_results[:,0], yaxis=tsne_results[:,1],kind=labels))
        plt.figure(figsize=(7,7))
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
        h,l=g.get_legend_handles_labels()
        n=len(set(df['kind'].values.tolist()))
        plt.legend(h[0:n+1],l[0:n+1])
        plt.tight_layout()
        plt.savefig('kinome_ESM_tsne_%s_%d.pdf'%(m,perp),dpi=300)
