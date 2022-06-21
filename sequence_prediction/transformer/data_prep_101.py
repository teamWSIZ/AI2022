from my_work.torch_adv.word_language_model import data

corpus = data.Corpus('data/lalka')

corpus.dictionary.add_word('wogóle')  # tego nie było w tekście Lalki
sentence = 'co to wogóle ma znaczyć'
words = sentence.split()
ids = []
for word in words:
    ids.append(corpus.dictionary.word2idx[word])
print(ids)

print('-' * 80)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to('cpu')


print(corpus.test.data[:60])

# batches: cut to sequences, batch#0: 0th words, batch#2: 1st words...
# prepared for feeding into the network batch#0, then batch#1, batch#2 etc..
# if network needs to get sequences in order, the only way to "batch-speed-it-up" is by
# evaluating very many sequences at once
#
# example: batchify(print(string.ascii_lowercase), 4):
# ┌ a g m s ┐ # batch#0
# │ b h n t │ # batch#1
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
#
gg = batchify(corpus.test, bsz=10)
print(gg[:, 0])  # all elements at [0] of all batches → sequence of consecutive words from the text (at the beginning)
print(gg.size())
