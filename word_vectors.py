# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
from scipy import spatial


def main():
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # https://radimrehurek.com/gensim/models/keyedvectors.html
    word_vectors = model.wv
    print('woman is to king, then man is to:', word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))

    print('Similarity between dog and dogs:', word_vectors.similarity('dog', 'dogs'))

    # http://www.cs.toronto.edu/~yangxu/csc2604w18_lab1_extension.pdf
    result = 1 - spatial.distance.cosine(model['dog'], model['dogs'])
    result1 = 1 - spatial.distance.cosine(model['cat'], model['cats'])
    print(result)
    print(result1)


if __name__ == "__main__":
    main()
