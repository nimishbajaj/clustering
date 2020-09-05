from flask import Flask, request, json
import cluster

app = Flask(__name__)


@app.route('/')
def home():
    return "Service is up!"


@app.route('/cluster', methods=['POST'])
def predict():
    keyWordDict = request.json.get('dict')
    parsedKeyWordDict = {}
    word_vectors = []
    keyTerms = []
    for x in keyWordDict:
        word_vectors.append([float(x) for x in keyWordDict[x][1]])
        parsedKeyWordDict[x] = keyWordDict[x][0]
        keyTerms.append(x)
        # print(keyWordDict[x])
    # print(word_vectors)
    # print(parsedKeyWordDict)
    return cluster.get_tags(keyTerms, parsedKeyWordDict, word_vectors)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
