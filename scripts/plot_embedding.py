import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    requiredArguments = parser.add_argument_group('required arguments')
    requiredArguments.add_argument('-e', '--embedding', help='Pickle file with the'
                                   ' embeddings', required=True)
    parser.add_argument('-p', '--plot', help='Type of plot. Either'
                        ' t-SNE or UMAP', default='t-SNE')
    parser.add_argument('-l', '--labels', help='Labels for the embeddings',
                         default=None)
    parser.add_argument('-o', '--outname', help='Plots outnames default t-SNE/UMAP')
    parser.add_argument('--per', help='Perplexity values for t-SNE'
                        ' default list with 10 15 20 30 40', nargs='+', type=int,
                        default=[10, 15, 20, 30, 40])
    parser.add_argument('--mind', help = 'Minimum distances values for'
                        ' UMAP default list with 0.0 0.1 0.25 0.5 0.75 0.99', nargs='+',
                        type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.99])
    parser.add_argument('--neig', help = 'Neighbours values for UMAP'
                         ' default list with 5 15 30 50 75 100', nargs='+', type=int,
                         default=[5, 15, 30, 50, 75, 100])
    parser.add_argument('--met', help = 'Metrics default list with'
                          'euclidean l2 braycurtis cosine canberra ...', nargs='+',
                          default=['l2', 'braycurtis', 'correlation', 'l1',
                                   'manhattan', 'euclidean', 'cityblock' ,
                                   'minkowski','sqeuclidean', 'cosine', 'minkowski',
                                   'nan_euclidean', 'canberra'])

    args = parser.parse_args()
    embedding = args.embedding
    plot = args.plot
    perplexities = args.per
    mindists = args.mind
    neighbours = args.neig
    metrics = args.met
    outname = args.outname
    labels = args.labels

    with open(embedding, 'rb') as input_file:
        embeddings = pickle.load(input_file)

    if labels is not None:
        with open(labels, 'rb') as input_file:
            labels = pickle.load(input_file)

    embeddings = np.asarray(embeddings)

    if plot == 't-SNE':
        from sklearn.manifold import TSNE
        for perp in perplexities:
            for m in metrics:
                tsne=TSNE(n_components=2, verbose=1, learning_rate='auto', init='pca',
                          perplexity=perp, n_iter=1500, early_exaggeration=12, metric=m)
                tsne_results=tsne.fit_transform(embeddings)
                plt.figure(figsize=(7,7))
                if labels is not None:
                    df=pd.DataFrame(dict(xaxis=tsne_results[:,0],
                                         yaxis=tsne_results[:,1],kind=labels))
                    g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
                    h,l=g.get_legend_handles_labels()
                    n=len(set(df['kind'].values.tolist()))
                    plt.legend(h[0:n+1],l[0:n+1])
                else:
                    df=pd.DataFrame(dict(xaxis=tsne_results[:,0],yaxis=tsne_results[:,1]))
                    g=sns.scatterplot(data=df, x='xaxis', y='yaxis')
                plt.tight_layout()
                plt.savefig('tSNE_%s_%s_%d.pdf' % (outname, m, perp),dpi=300)
    if plot == 'UMAP':
        import umap as mp
        for mindist in mindists:
            for neigh in neighbours:
                umap=mp.UMAP(n_neighbors=neigh, n_epochs=1000, min_dist=mindist,
                             metric = 'euclidean')
                UMAP_results=umap.fit_transform(embeddings)
                plt.figure(figsize=(7,7))
                if labels is not None:
                    df=pd.DataFrame(dict(xaxis=UMAP_results[:,0],
                                         yaxis=UMAP_results[:,1], kind=labels))
                    g=sns.scatterplot(data=df, x='xaxis', y='yaxis',hue='kind')
                    h,l=g.get_legend_handles_labels()
                    n=len(set(df['kind'].values.tolist()))
                    plt.legend(h[0:n+1],l[0:n+1])
                else:
                    df=pd.DataFrame(dict(xaxis=UMAP_results[:,0], yaxis=UMAP_results[:,1]))
                    g=sns.scatterplot(data=df, x='xaxis', y='yaxis')
                plt.tight_layout()
                plt.savefig('UMAP_%s_%s_%d.pdf' % (outname, m, perp),dpi=300)
