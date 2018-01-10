import subprocess
import sys
import numpy
import sklearn.cluster as clus
from sklearn.decomposition import PCA
from collections import Counter
import codecs
import os
import argparse


class ClusterCurr:

    def __init__(self):
        self.numClusters = 1
        self.clustSource = []
        self.clustTarget = []
        self.clustScorFiles = []
        self.clusSizes = []

    def genSent2vecs(self, inputFile, outputFile, vecSize, epochs=10):

        print("Input file: "+inputFile)

        import multiprocessing

        threads = multiprocessing.cpu_count()

        # Generate binary model
        command = "./fasttext sent2vec -input {0} " \
                  "-output {1} -dim {2} -epoch {3} -thread {4}".format(inputFile, outputFile,
                                                           vecSize, epochs, threads)
        subprocess.call(command, shell=True)

        # Output sentence scores from binary model to text file
        command = "./fasttext print-sentence-vectors {0}.bin < {1}" \
                  " > {2}".format(outputFile, inputFile, outputFile)

        subprocess.call(command, shell=True)
        os.remove(outputFile+".bin")

        print("Generated txt model.")

        return outputFile

    def loadAndPCA(self, file, pcaInp=False):

        x = numpy.loadtxt(file)

        if pcaInp:
            print("PCAing..")
            reduct = PCA(n_components='mle', svd_solver="full")
            xp = reduct.fit_transform(x)
            print("PCA done..")
            print("Vector length after PCA: {0}".format(len(xp[0])))
            return xp

        return x

    def clusterKmeans(self, file, numClus, pca=False):


        print("Clustering...")
        x = self.loadAndPCA(file, pca)

        self.numClusters = numClus

        # Check nltk clustering with cosine distance

        clusterer = clus.MiniBatchKMeans(numClus, verbose=True, batch_size=5000,
                                         max_no_improvement=1000, compute_labels=True,
                                         reassignment_ratio=0.001)
        #clusterer = clus.KMeans(n_clusters=numClus, n_jobs=-1, verbose=1)
        scores = clusterer.fit_transform(x)
        print("Clustering done.")

        counts = Counter(clusterer.labels_)

        # Add counts
        for i in range(0, len(counts)):
            self.clusSizes.append(counts[i])

        print("Clustering output: ")
        print(self.clusSizes)

        # TODO : Check the outcome of clustering from different
        # Embedding sizes and with/without PCA

        return clusterer.labels_, scores

    def clustertoFiles(self, sourceSents, targetSents, numClusters, clustLabs, clustScores):

        # Create new parallel files for each cluster of sentences
        # and write their respective scores to files as well.
        import os

        src_dir = sourceSents + "_src_clus"
        os.makedirs(src_dir)
        src_dir += "/"

        tgt_dir = targetSents + "_tgt_clus"
        os.makedirs(tgt_dir)
        tgt_dir += "/"

        score_dir = sourceSents + "_score_clus"
        os.makedirs(score_dir)
        score_dir += "/"

        clustScorFiles = []
        clustSource = []
        clustTarget = []

        # Save file names
        for i in range(0, numClusters):
            self.clustSource.append("{0}clus{1}_{2}".format(src_dir, i, sourceSents))
            self.clustTarget.append("{0}clus{1}_{2}".format(tgt_dir, i, targetSents))
            self.clustScorFiles.append("{0}score{1}_{2}".format(score_dir, i, sourceSents))

        # Write the file names in the output while opening file handles
        with codecs.open("files_" + sourceSents, 'w', encoding='utf-8') as outFiles:
            for i in range(0, numClusters):
                clustSource.append(codecs.open(self.clustSource[i], mode='w', encoding="utf-8"))
                clustTarget.append(codecs.open(self.clustTarget[i], mode='w',encoding="utf-8"))
                clustScorFiles.append(codecs.open(self.clustScorFiles[i], mode='w', encoding="utf-8"))
                outFiles.write(self.clustSource[i]+"\r\n")
                outFiles.write(self.clustTarget[i]+"\r\n")

        with codecs.open(sourceSents, encoding='utf-8') as f:
            sourceLines = f.readlines()
        with codecs.open(targetSents, encoding='utf-8') as f:
            targetLines = f.readlines()

        for i in range(0, len(sourceLines)):
            # use the cluster label as index to which file to write to
            clustSource[clustLabs[i]].write(sourceLines[i])
            clustTarget[clustLabs[i]].write(targetLines[i])
            # Just add the score of the cluster it belongs to (smaller = closer)
            clustScorFiles[clustLabs[i]].write(str(clustScores[i][clustLabs[i]])+"\r\n")

        for i in range(0, numClusters):
            clustSource[i].close()
            clustTarget[i].close()
            clustScorFiles[i].close()

        return 1

    def genClustFiles(self, vecFile="", src="", trgt="", vecSize=100, numClusts=3, pca=False):

        # Parallel texts
        source = src
        target = trgt
        # Size of the sentence vector
        sentVecSize = vecSize

        # If vectors file not given, generate it.
        if not vecFile:
            # Model file name
            vecFile = "sentVecs_{0}_{1}".format(sentVecSize,source)

            # Generate sent vectors for source language
            # We assume here semantic similarity in both the
            # spaces of the source and target
            vecFile = self.genSent2vecs(source, vecFile, sentVecSize)

        # Number of Clusters to generate
        numClusters = numClusts
        # PCA (or other Dim reduction) before clustering
        doPCA = pca

        labels, scores = self.clusterKmeans(vecFile, numClusters)

        self.clustertoFiles(source, target, numClusters, labels, scores)

        return vecFile

    def sortClusters(self, asc=True):

        # Sorts cluster files in place

        sourceLines= []
        targetLines = []
        scores = []

        for i in range(0, len(self.clustScorFiles)):
            with codecs.open(self.clustSource[i], encoding='utf-8') as s:
                sourceLines = s.readlines()
            with codecs.open(self.clustTarget[i], encoding='utf-8') as t:
                targetLines = t.readlines()
            scores = numpy.loadtxt(self.clustScorFiles[i])

            # Zip, sort, unzip
            all = zip(scores, sourceLines, targetLines)

            if asc:
                all = sorted(all, key=lambda x: x[0])
            else:
                all = sorted(all, reverse=True, key=lambda x: x[0])

            scores, sourceLines, targetLines = zip(*all)

            src = codecs.open(self.clustSource[i], mode='w', encoding="utf-8")
            src.writelines(sourceLines)
            trgt = codecs.open(self.clustTarget[i], mode='w', encoding="utf-8")
            trgt.writelines(targetLines)
            scor = codecs.open(self.clustScorFiles[i], mode='w', encoding="utf-8")
            scor.writelines([str(x)+"\r\n" for x in scores])
            src.close()
            trgt.close()
            scor.close()

    def batchToFiles(self, src, trgt, finalIndices):

        clustSource = []
        clustTarget = []
        srcOut = codecs.open(src, 'w', encoding="utf-8")
        trgtOut = codecs.open(trgt, 'w', encoding="utf-8")
        currentIndices = []

        for i in range(0, self.numClusters):
            with codecs.open(self.clustSource[i], 'r', encoding="utf-8") as sr:
                clustSource.append(sr.readlines())
            with codecs.open(self.clustTarget[i], 'r', encoding="utf-8") as tr:
                clustTarget.append(tr.readlines())
            currentIndices.append(0)

        for sample in finalIndices:
            srcOut.write(clustSource[sample][currentIndices[sample]])
            trgtOut.write(clustTarget[sample][currentIndices[sample]])
            currentIndices[sample] += 1

        # close files
        srcOut.close()
        trgtOut.close()

    def batcher(self, batchSize = 64, normalize=False, outputName=""):
        # This function builds the final file by collecting sentences
        # From the cluster files, as per the probab of each cluster
        # according to its size, and to the required batch size.

        currentClusSizes = self.clusSizes[:]

        # Update sizes and probabilities as we progress
        # by decreasing the already selected

        finalOut = []

        totalSize = float(sum(currentClusSizes))
        while totalSize:

            # We play HERE!
            currentBatch = batchSize if totalSize > batchSize else totalSize

            if not normalize:
                probs = [prob / totalSize for prob in currentClusSizes]
                # Uses probabs based on size
                batchIndices = numpy.random.choice(self.numClusters, size=batchSize,
                                               replace=True, p=probs)
            else:
                # Make probabs equal, regardless of size
                # If size is zero
                choices = []
                for size in currentClusSizes:
                    if size:
                        choices.append(1)
                    else:
                        choices.append(0)
                probs = [choice / float(sum(choices)) for choice in choices]  # Uses probabs based on size
                batchIndices = numpy.random.choice(self.numClusters, size=batchSize,
                                        replace=True, p=probs)
            #print(probs)
            #print(batchIndices)

            batchStats = Counter(batchIndices)
            print(batchStats)

            for sampleClus in batchIndices:
                if currentClusSizes[sampleClus]:
                    finalOut.append(sampleClus)
                    currentClusSizes[sampleClus] -= 1
                else:
                    # Empty, pick another cluster randomly
                    totalSize = float(sum(currentClusSizes))

                    if totalSize:
                        if not normalize:
                            # Probabs according to sizes
                            probs = [prob / totalSize for prob in currentClusSizes]
                            # Uses probabs based on size
                            i = numpy.random.choice(self.numClusters,
                                                               replace=True, p=probs)
                            if currentClusSizes[i]:
                                finalOut.append(i)
                                currentClusSizes[i] -= 1
                        else:
                            # Probabs equal to all non empty sizes
                            choices = []
                            for size in currentClusSizes:
                                if size:
                                    choices.append(1)
                                else:
                                    choices.append(0)
                            probs = [choice/float(sum(choices)) for choice in choices]                        # Uses probabs based on size
                            i = numpy.random.choice(self.numClusters,
                                                    replace=True, p=probs)
                            if currentClusSizes[i]:
                                finalOut.append(i)
                                currentClusSizes[i] -= 1

            totalSize = float(sum(currentClusSizes))
            print(currentClusSizes)

        srcOut = "src_" + outputName
        trgtOut = "trgt_" + outputName
        self.batchToFiles(srcOut, trgtOut, finalOut)

        return 1

    def getSegments(self, scores, simCutOff, harsh=True):
        # difference between each item and the one before it

        difScores = [1]
        # Differences are shifted by one
        difScores.extend(numpy.diff(scores))
        scores = numpy.asarray(difScores)

        # differences smaller than simCutOff indicate similarity
        # mark it zero and the one before it which caused it.
        for j in range(1, len(scores)):
            if scores[j] < simCutOff:
                scores[j] = 0
                scores[j - 1] = 0

        # Consecutive zeros can all be replaced by one of them
        # Cause they are close enough (harsh mode)
        segments = []

        if harsh:
            looper = 0
            while looper < len(scores):
                if scores[looper] == 0:
                    startSeg = looper
                    while looper < len(scores) and scores[looper] == 0:
                        looper += 1
                    segments.append((startSeg, looper - 1))
                looper += 1
        else:
            # pairs of points , but take the first one
            # To avoid off by one error
            looper = 0
            while looper < len(scores):
                if scores[looper] == 0:
                    startSeg = looper
                    looper += 1
                    segments.append((startSeg, looper))
                looper += 1

        print("Segments:")
        print(segments)

        return segments

    def cutOffSents(self, simCutOff=0.000001, harsh=True):

        # pass on the arranged array, any 2 items that differ by
        # edge detection!!!!!!!!!!!! and the areas in the middle
        # can be replaced by one value picked randomly

        # Sorts cluster files in place

        newSize = self.clusSizes[:]

        for i in range(0, len(self.clustScorFiles)):
            with codecs.open(self.clustSource[i], encoding='utf-8') as s:
                sourceLines = s.readlines()
            with codecs.open(self.clustTarget[i], encoding='utf-8') as t:
                targetLines = t.readlines()
            scores = numpy.loadtxt(self.clustScorFiles[i])

            segments = self.getSegments(scores, simCutOff, harsh)

            # Every 2 consecutive zeros can be replaced by 1 of them
            # To avoid build up of errors (less harsh)

            # Actual removal happens here (with option so people can
            # first experiment with the hyperparameter and see.
            # Add the dead to a new file to be used later maybe

            # BEWARE: in place removal will mess up the
            # Indexing

            #print(scores)
            # print(len(scores))
            # print(len(scores) - numpy.count_nonzero(scores))
            # indices = numpy.where(scores == 0)[0]
            # print(len(indices))
            # print(indices)

            # Zip, sort, unzip
            # Add new cluster size
            # Go through segments and zero out the
            # Sentences, they won't get written
            # in the output
            if harsh:
                for segment in segments:
                    for x in range(segment[0], segment[1]):
                        sourceLines[x] = ""
                        targetLines[x] = ""
                        newSize[i] -= 1

            else:
                for segment in segments:
                    sourceLines[segment[0]] = ""
                    targetLines[segment[0]] = ""
                    newSize[i] -= 1

            # Write final output
            src = codecs.open(self.clustSource[i], 'w', encoding='utf-8')
            src.writelines(sourceLines)
            trgt = codecs.open(self.clustTarget[i], 'w', encoding='utf-8')
            trgt.writelines(targetLines)

            src.close()
            trgt.close()

        # New sizes vs old
        oldTotalSize = sum(self.clusSizes)
        newTotalSize = sum(newSize)

        print("Removed {0} from {1} sentences.".format(oldTotalSize-newTotalSize, oldTotalSize))
        print("Reduction to {:.3%} of original size.".format(newTotalSize/oldTotalSize))

        # write new sizes
        self.clusSizes = newSize

        return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ClusterReg')
    parser.add_argument('--src', help='Source sentences', default="src-train.txt")
    parser.add_argument('--tgt', help='Target sentences', default="tgt-train.txt")
    parser.add_argument('--K', help='K in KMeans', type=int, default=64)
    parser.add_argument('--vec', help='Sentence embeddings size', type=int, default=100)
    parser.add_argument('--embeds', help='Embeddings file', default="")
    parser.add_argument('--pca', help='Use PCA before clustering', type=int, default=1)
    parser.add_argument('--asc', help='Sort ascending', type=int, default=1)
    parser.add_argument('--batch', help='Batch size', type=int, default=64)
    parser.add_argument('--normalize', help='Batch by normalized or groupReg', type=int, default=1)
    parser.add_argument('--out', help='Output file prefix', default="batched.txt")

    # Cutoff params
    parser.add_argument('--cutoff', help='Cutoff threshold', type=float, default=0.0001)
    parser.add_argument('--harsh', help='cutoff type', type=int, default=0)

    args = parser.parse_args()
    print(args)

    # "sentVecs_5_src-train.txt"
    groupReg = ClusterCurr()
    vecFile = groupReg.genClustFiles(vecFile=args.embeds,
                                     src=args.src, trgt=args.tgt, vecSize=args.vec,
                                     numClusts=args.K, pca=args.pca)
    groupReg.sortClusters(asc=args.asc)
    #groupReg.cutOffSents(harsh=args.harsh)
    groupReg.batcher(batchSize=args.batch, normalize=args.normalize, outputName=args.out)
