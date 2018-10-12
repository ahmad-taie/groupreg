# Groupreg

The code requires the sent2vec binary (fasttext) to be in the same directory.

To run the script:

```
python groupReg.py --src corpus.src --tgt corpus.tgt
```

Optional parameters:
```
    --K, number of clusters, default = 64
    --vec, Sentence embeddings size, default = 100
    --embeds, Sentence embeddings file
    --batch, mini-batch size, default = 64
    --out, Output files prefix, default="batched"
    --normalize, Uses the GroupReg probabilities, default = True (if false, uses probabilities according to cluster sizes "non-normalized")
```
