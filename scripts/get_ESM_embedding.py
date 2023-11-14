import time
import torch
import esm
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_tensor_shape(tensor, device):
    if device == 'cpu':
        shape = tensor.numpy().shape
    else:
        shape = tensor.cpu().detach().numpy().shape
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
        return seqs, titles, None

def select_device():
    free, total = torch.cuda.mem_get_info(device='cuda:0')
    perc0 =  ((total - free) / total) * 100
    free, total = torch.cuda.mem_get_info(device='cuda:1')
    perc1 =  ((total - free) / total) * 100
    print('GPU 0: %.2f%%' % perc0)
    print('GPU 1: %.2f%%' % perc1)
    if perc0 > perc1:
        return "cuda:1"
    else: return "cuda:0"

def get_GPU_memory(device):
    current_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    max_reserved_memory = torch.cuda.max_memory_reserved(device)
    print('.........GPU Memory.........')
    print(f"Current GPU memory usage: {current_memory / 1024 / 1024:.2f} MB")
    print(f"Reserved GPU memory: {reserved_memory / 1024 / 1024:.2f} MB")
    print(f"Max Reserved GPU memory: {max_reserved_memory / 1024 / 1024:.2f} MB")
    free, total = torch.cuda.mem_get_info(device)
    print(f'Free {free / 1024 / 1024:.3f} MB ')
    print(f'Total {total / 1024 / 1024:.3f} MB ')
    print('...........................')

def plot_tSNEs(embeddings, outname, labels = None,
               metric='euclidean', perplexity=30, n_iter=1500):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, verbose=1, learning_rate='auto', init='pca',
                perplexity=perplexity, n_iter=n_iter, early_exaggeration=12,
                metric=metric)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize = (7, 7))
    if labels is not None:
        df = pd.DataFrame(dict(xaxis=tsne_results[:, 0],
                               yaxis=tsne_results[:, 1], kind=labels))
        g = sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
        h, l = g.get_legend_handles_labels()
        n = len(set(df['kind'].values.tolist()))
        plt.legend(h[0:n+1], l[0:n+1])
    else:
        df = pd.DataFrame(dict(xaxis=tsne_results[:, 0],
                               yaxis=tsne_results[:, 1]))
        g = sns.scatterplot(data=df, x='xaxis', y='yaxis')
    plt.tight_layout()
    plt.savefig('%s_tsne_%s_%d.pdf' % (outname, metric, perplexity), dpi=300)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('-f', '--fasta', help='Fasta file',
                                   required=True)
    requiredArguments.add_argument('-o', '--out', help='Outname',
                                   required=True)
    parser.add_argument('-b', '--batch', help='Batch size',
                        default=10)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true',
                        default=False)
    parser.add_argument('-m', '--model', help='ESM model it can be 650M or 3B',
                        default='650M')
    parser.add_argument('-d', '--deli', help='Delimiter of the fasta file labels',
                        default=None)

    args = parser.parse_args()
    verbose = args.verbose
    fasta = args.fasta
    batch_size = int(args.batch)
    delimiter = args.deli
    model = args.model
    outname = args.out

    # Select CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device = select_device()
    if verbose:
        print(device)
        get_GPU_memory(device=device)

    seqs, titles, labels = read_fasta(fasta, unalign=True, delimiter=delimiter)
    data = [(titles[i], seqs[i]) for i in range(len(seqs))]

    # Load ESM-2 model
    print('Loading ESM-2 %s model' % model)
    if model == '650M':
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        layers = 33
    elif model == '3B':
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        layers = 36
    elif model == '15B':
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        layers = 48
    else:
        raise ValueError('Model must be either 650M or 3B')
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    if verbose:
        print('After loading the model')
        get_GPU_memory(device=device)

    # Get the sequence embeddings
    start_time = time.time()

    print('Embedding the sequences')
    seqs_embeddings = []
    for i in range(0, len(data), batch_size):
        print('%d / %d' % (i, len(data)))
        batch = data[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(device)
        if verbose:
            print('After tokenizing')
            get_GPU_memory(device=device)
            #print(batch_labels)
            #print(batch_strs)
            #print(batch_tokens)
            shape = get_tensor_shape(batch_tokens, device)
            print(shape)

        # Extract per-residue representations
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layers], return_contacts=False)
        token_representations = results["representations"][layers]
        if verbose:
            print('After getting the embedding')
            get_GPU_memory(device=device)
            #print(token_representations)
            shape = get_tensor_shape(token_representations, device)
            print(shape)

        # Generate per-sequence representations via averaging
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            seq_token_representations = token_representations[i, 1 : tokens_len - 1].mean(0)
            if 'cuda' in device:
                seq_token_representations = seq_token_representations.to('cpu')
            sequence_representations.append(seq_token_representations)
        if verbose:
            shape = get_tensor_shape(sequence_representations[0], device)
            print(shape)
        for sequence_representation in sequence_representations:
            seqs_embeddings.append(sequence_representation)

    end_time = time.time()
    execution_time = end_time - start_time

    with open('%s_embeddings.pickle' % outname, 'wb') as handle:
        pickle.dump(seqs_embeddings, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    if delimiter is not None:
        with open('%s_labels.pickle' % outname, 'wb') as handle:
            pickle.dump(labels, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


    print("---------Finished embedding the sequences---------")
    print('Execution time: %.2f s' % execution_time)


    #perps = [10,15,20,30,40]
    #metrics = ['l2', 'braycurtis', 'correlation', 'l1', 'manhattan', 'euclidean', 'cityblock' , 'minkowski','sqeuclidean', 'cosine', 'minkowski', 'nan_euclidean', 'canberra']
