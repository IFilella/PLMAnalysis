import random

genes = ['TssA','TssB','TssC','TssE','TssF','TssG','TssJ','TssK','TssL','TssM']

T6SS_shuffle = []
for gene in genes:
    print(gene)
    f = open('../data/%s.fasta'%gene,'r')
    seqs = []
    titles = []
    for line in f:
        if line[0] == '>': titles.append(gene+'_' + line.replace('\n','').replace('>',''))
        else: seqs.append(line.replace('\n','').replace('-',''))
    rand = [random.randint(0, len(seqs)) for i in range(30)]
    rand_seqs = [(titles[i],seqs[i]) for i in rand]
    T6SS_shuffle = T6SS_shuffle + rand_seqs

fout = open('../data/T6SS_shuffle.fasta','w')
for seq in T6SS_shuffle:
    fout.write('>%s\n%s\n'%(seq[0],seq[1]))

