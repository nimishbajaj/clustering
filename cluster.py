from sklearn.cluster import Birch


def get_tags(keyTerms, keyWordToScore, word_vectors):
    NUM_CLUSTERS = 8
    kmeans = Birch(n_clusters=NUM_CLUSTERS)
    kmeans.fit(word_vectors)
    labels = kmeans.labels_
    findLabel = lambda x: max(x, key=(lambda key: keyWordToScore[key]))

    output = {}
    for x, y in zip(labels, keyTerms):
        if x in output:
            output[x].append(y)
        else:
            output[x] = [y]

    result = {}
    for k, v in output.items():
        topLabel = findLabel(v)
        result[topLabel] = round(len(v)/len(keyTerms)*100,2)

    return result
