from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from MolecularAnalysis.moldb import MolDB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sets = np.genfromtxt('../data/selected_sets_names.txt',skip_header=0,delimiter=',',dtype=str)
setspath = '/data2/julia/sets/'

moldbs = []
for s in sets:
    moldbobj = MolDB(sdfDB='%s/%s.sdf' % (setspath, s), verbose=False)
    moldbs.append(moldbobj)

#model_name = 'seyonec/ChemBERTa-zinc-base-v1'
model_name = 'DeepChem/ChemBERTa-77M-MLM' #ChemBERTA-2
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True).to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name)

labels = []
embeddings = []
for i,moldbobj in enumerate(moldbs):
    smiles = moldbobj.smiles
    for j,smile in enumerate(smiles):
        input_token = tokenizer(smile, return_tensors='pt').to('cuda:0')
        output = model(**input_token, output_hidden_states=True)
        embedding = output.hidden_states[0]
        embedding = embedding.cpu().detach().numpy()
        print(embedding.shape)
        exit()
        embedding = np.squeeze(embedding)
        embedding = embedding.mean(axis=0)
        embeddings.append(embedding)
    labels.extend([sets[i]]*len(smiles))

embeddings = np.asarray(embeddings)
from sklearn.manifold import TSNE
perps = [10,15,20,30,40]
metrics = ['l2', 'braycurtis', 'correlation', 'l1', 'manhattan','euclidean', 'cityblock', 'minkowski', 'sqeuclidean', 'cosine', 'minkowski', 'nan_euclidean','canberra']
for perp in perps:
    for m in metrics:
        tsne=TSNE(n_components=2, verbose=1,learning_rate='auto', init='pca',
                  perplexity=perp, n_iter=1500,early_exaggeration=12, metric=m)
        tsne_results=tsne.fit_transform(embeddings)
        df=pd.DataFrame(dict(xaxis=tsne_results[:,0],yaxis=tsne_results[:,1], kind=labels))
        plt.figure(figsize=(7,7))
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
        h,l=g.get_legend_handles_labels()
        n=len(set(df['kind'].values.tolist()))
        plt.legend(h[0:n+1],l[0:n+1])
        plt.tight_layout()
        plt.savefig('molecules_chemBERTa2_tsne_%s_%d.pdf'%(m,perp),dpi=300)

import umap as mp
min_dists = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
neighbours = [5, 10, 15, 25, 40]
for mindist in min_dists:
    for neigh in neighbours:
        umap=mp.UMAP(n_neighbors=neigh, n_epochs=1000, min_dist=mindist, metric = 'euclidean')
        UMAP_results=umap.fit_transform(embeddings)
        df=pd.DataFrame(dict(xaxis=UMAP_results[:,0],yaxis=UMAP_results[:,1], kind=labels))
        plt.figure(figsize=(7,7))
        g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
        h,l=g.get_legend_handles_labels()
        n=len(set(df['kind'].values.tolist()))
        plt.legend(h[0:n+1],l[0:n+1])
        plt.tight_layout()
        plt.savefig('molecules_chemBERTa2_UMAP_%.2f_%d.pdf'%(mindist, neigh),dpi=300)

