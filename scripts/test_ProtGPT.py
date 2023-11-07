from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Human Kinome
#msa = '../data/T6SS_shuffle.fasta'
msa = '../data/human_kinome_noPLK5.aln'
f = open(msa,'r')
seqs = []
titles = []
for line in f:
    if line[0] == '>': titles.append(line.replace('\n','').         replace('>',''))
    else: seqs.append(line.replace('\n','').replace('-',''))
data = [(titles[i],seqs[i]) for i in range(len(seqs))]
batch_size = 2
labels = [title.split('_')[0] for title in titles]

"""
model_name = "nferruz/ProtGPT2"
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name)

total_embeddings = []
for i,seq in enumerate(seqs):
    print(i)
    inputs = tokenizer(seq, return_tensors="pt").to('cuda:0')
    outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[0]
    embeddings = embeddings.cpu().detach().numpy()
    embeddings = np.squeeze(embeddings, axis=0)
    print(embeddings.shape)
    total_embeddings.append(embeddings)

#with open('../results/T6SS_shuffle_ProtGPT.pickle', 'wb') as handle:
with open('../results/human_kinome_ProtGPT.pickle', 'wb') as handle:
    pickle.dump(total_embeddings, handle,
                protocol=pickle.HIGHEST_PROTOCOL)
"""
with open("../results/human_kinome_ProtGPT.pickle", "rb") as input_file:
#with open("../results/T6SS_shuffle_ProtGPT.pickle", "rb") as input_file:
    total_embeddings = pickle.load(input_file)

total_embeddings2 = []
for emb in total_embeddings:
    #print(emb.shape)
    emb = emb.mean(0)
    #print(emb.shape)
    total_embeddings2.append(emb)
total_embeddings2 = np.asarray(total_embeddings2)
print(total_embeddings2.shape)

from sklearn.manifold import TSNE
perps = [10,15,20,30]
metrics = ['l2', 'braycurtis', 'correlation', 'l1', 'manhattan', 'euclidean', 'cityblock' , 'minkowski',         'sqeuclidean', 'cosine', 'minkowski', 'nan_euclidean', 'canberra']
for perp in perps:
    for m in metrics:
        tsne=TSNE(n_components=2, verbose=1, learning_rate='auto', init='pca',
                  perplexity=perp, n_iter=1500, early_exaggeration=12, metric=m)
        tsne_results=tsne.fit_transform(total_embeddings2)
        df=pd.DataFrame(dict(xaxis=tsne_results[:,0], yaxis=tsne_results[:,1],kind=labels))
        plt.figure(figsize=(7,7))
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
        h,l=g.get_legend_handles_labels()
        n=len(set(df['kind'].values.tolist()))
        plt.legend(h[0:n+1],l[0:n+1])
        plt.tight_layout()
        plt.savefig('T6SS_ProtGTP_tsne_%s_%d.pdf'%(m,perp),dpi=300)
