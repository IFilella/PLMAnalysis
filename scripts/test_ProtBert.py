from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read Fasta File and set paramaters
msa = '../data/human_kinome_noPLK5.aln'
#msa = '../data/T6SS_shuffle.fasta'
f = open(msa,'r')
seqs = []
titles = []
for line in f:
    if line[0] == '>': titles.append(line.replace('\n','').replace('>',''))
    else: seqs.append(line.replace('\n','').replace('-',''))
batch_size = 25
labels = [title.split('_')[0] for title in titles]
lengths = [len(seq) for seq in seqs]
seq_len = max(lengths)+2
print(seq_len)

#"""
# Load the pretrained ProtBert model with a given seq_len
pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))

encoded_x = input_encoder.encode_X(seqs,seq_len)
print(encoded_x[0], encoded_x[0].shape)
local_representation, global_representation = model.predict(encoded_x,batch_size=batch_size)
print(local_representation)
print(local_representation.shape)
#local_representation = local_representation.mean(1)
local_representation = local_representation.reshape(*local_representation.shape[:-2], -1)
print(local_representation)
print(local_representation.shape)


from sklearn.manifold import TSNE
perps = [10,15,20,30,40]
metrics = ['l2', 'braycurtis', 'correlation', 'l1', 'manhattan','euclidean', 'cityblock', 'minkowski','sqeuclidean', 'cosine', 'minkowski', 'nan_euclidean', 'canberra']

perps = [15]
metrics = ['canberra']
for perp in perps:
    for m in metrics:
        tsne=TSNE(n_components=2, verbose=1, learning_rate='auto', init='pca',
                  perplexity=perp, n_iter=1500, early_exaggeration=12, metric=m)
        tsne_results=tsne.fit_transform(local_representation)
        df=pd.DataFrame(dict(xaxis=tsne_results[:,0], yaxis=tsne_results[:,1], kind=labels))
        plt.figure(figsize=(7,7))
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
        h,l=g.get_legend_handles_labels()
        n=len(set(df['kind'].values.tolist()))
        plt.legend(h[0:n+1],l[0:n+1])
        plt.tight_layout()
        plt.savefig('Kinome_ProtBert_tsne_%s_%d.pdf'%(m,perp),dpi=300)
