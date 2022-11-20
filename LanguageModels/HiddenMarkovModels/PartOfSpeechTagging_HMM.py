import os
import sys
import time
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def parse_file(path):
    sentences = []
    with open(path,"r") as corpusFP:
        for line in corpusFP:
            sentence = []
            if "\n" == line:
                continue

            line = line.lower()

            for token in line.split(" "):
                try:
                    word, partOfSpeech = token.split("/")
                    sentence.append((word,partOfSpeech))
                except Exception as e:
                    continue
            sentences.append(sentence)
    return sentences

def load_corpus(path):
    """ Load corpus from a folder / directory

    Arg:
    path: a text sequence denotes the path of corpus

    Return:
    sentences: a list of sentences that are preprocessed in the corpus
    """
    sentences = []
    if not os.path.exists(path):
        print("Invalid path entered")
        return
    if os.path.isdir(path):
        for file in os.listdir(path):
            corpusPath = os.path.join(path,file)
            fileSentences = parse_file(corpusPath)
            sentences += fileSentences
    else:
        sentences = parse_file(path)
    return sentences

class HMMTagger:

    def __init__(self):
        """ Define variables that are used in the whole class

        You shuold initial all variables that are necessary and will be used
        globally in this class, such as the initial probability.
        """
        self.transitionProbabilities = None
        self.emissionProbabilities = None
        self.initialStateTransitions = None
        self.tag2index = {}
        self.word2index = {}
        self.numTags = 0
        self.numWords = 0

    def initialize_probabilities(self, sentences):
        """ Initialize / learn probabilities from the corpus

        In this function, you should learn inital probability, transition
        probability, and emission probability. Also, you should apply the
        add-one smoothing properly here.

        Arg:
            sentences: a list of sentences that are preprocessed in the corpus
        """
        for sentence in sentences:
            for word,tag in sentence:
                #assign words unique index
                if word not in self.word2index:
                    self.word2index[word] = self.numWords 
                    self.numWords = self.numWords  + 1
                else:
                    pass

                #assign tags unique index
                if tag not in self.tag2index:
                    self.tag2index[tag] = self.numTags
                    self.numTags = self.numTags + 1
                else:
                    pass

        emissionCounts =  np.zeros(shape=(self.numTags,self.numWords))
        transitionCounts =  np.zeros(shape=(self.numTags,self.numTags))
        transitionsFromInitialStateToTagCount = np.zeros(shape=(self.numTags,),dtype=np.int16)
        transitionsFromTagToEndStateCount = np.zeros(shape=(self.numTags,),dtype=np.int16)

        for sentence in sentences:
            for idx in range(0,len(sentence)):
                
                word = self.word2index[sentence[idx][0]]
                tag = self.tag2index[sentence[idx][1]]
        
                #track how many times we transition from START to tag
                if idx == 0:
                    transitionsFromInitialStateToTagCount[tag] = transitionsFromInitialStateToTagCount[tag] + 1

                #track how many times we transition from tag to END
                if idx == len(sentence) - 1:
                    transitionsFromTagToEndStateCount[tag] = transitionsFromTagToEndStateCount[tag] + 1

                #track how many times a tag t_i, occured after a tag t_i-1
                #Note: rows represent conditioning event (P( ti | t_i-1))
                if idx  > 0:
                    previousTag = self.tag2index[sentence[idx-1][1]]
                    transitionCounts[previousTag][tag] = transitionCounts[previousTag][tag] + 1

                #track how many times a tag t_i is associated with a word w_i
                emissionCounts[tag][word] = emissionCounts[tag][word] + 1

        #calculate transition probabilities
        transitionSums = transitionCounts.sum(axis=1)
        cols = transitionSums.shape[0]
        transitionSums = transitionSums.reshape((cols,1))
        self.transitionProbabilities = (transitionCounts + 1 )/ (transitionSums + self.numTags)

        #calculate emission probabilities
        emissionSums = emissionCounts.sum(axis=1)
        cols = emissionSums.shape[0]
        emissionSums = emissionSums.reshape((cols,1))
        self.emissionProbabilities = emissionCounts / emissionSums

        #calculate initial state transition probabilities
        self.initialStateTransitions = transitionsFromInitialStateToTagCount / sum(transitionsFromInitialStateToTagCount)
             
        #calculate end state transition probabilities
        self.endStateTransitions = transitionsFromTagToEndStateCount / sum(transitionsFromTagToEndStateCount)

    def viterbi_decode(self, sentence):
        """ Viterbi decoding algorithm implementation

        Arg:
            sentence: a text sequence needed to be decoded
        """
        try:
            sentence = sentence.split(' ')

            for word in sentence:
                if word not in self.word2index:
                    print('Word {} not present in corpus'.format(word))
                    raise Exception
            
            viterbi = np.zeros(shape=(len(self.tag2index),len(sentence)))
            backpointer = np.zeros(shape=(len(self.tag2index),len(sentence)))

            for tag in range(0,len(self.tag2index)):
                word = self.word2index[sentence[0]] #get first word in sentence
                viterbi[tag][0] = self.initialStateTransitions[tag] * self.emissionProbabilities[tag][word]
                backpointer[tag][0] = 0

            for word in range(1,len(sentence)):
                for tag in range(0,len(self.tag2index)):
                    previousWord = word - 1
                    paths = [viterbi[i][previousWord] * self.transitionProbabilities[i][tag] for i in range(0,self.numTags)]
                    viterbi[tag][word] = np.max(paths) * self.emissionProbabilities[tag][self.word2index[sentence[word]]]
                    backpointer[tag][word] = np.argmax(paths)

            index2tag = { idx : tag for tag, idx in self.tag2index.items()}
            bestPath = [index2tag[i].upper() for i in np.argmax(viterbi,axis=0)]

            return bestPath
        except:
            return None

if __name__ == "__main__":
    # Initialize the tagger class
    tagger = HMMTagger()

    # Read a corpus and learn from it
    folder_name = input("Input path: ")
    sentences = load_corpus(folder_name)
    
    tic = time.perf_counter()
    tagger.initialize_probabilities(sentences)
    toc = time.perf_counter()

    print(f"Time to initialize probabilities = {toc - tic:0.4f}")

    # Test
    sentence = "the planet jupiter and its moons are in effect a mini solar system ."
    result = tagger.viterbi_decode(sentence)
    print(sentence)
    print(result)
    print('\n')

    sentence = "computers process programs accurately ."
    result = tagger.viterbi_decode(sentence)
    print(sentence)
    print(result)
    print('\n')

    sentence = "the secretariat is expected to race tomorrow ."
    result = tagger.viterbi_decode(sentence)
    print(sentence)
    print(result)
    print('\n')

    sentence = "people continue to enquire the reason for the race to outer space ."
    result = tagger.viterbi_decode(sentence)
    print(sentence)
    print(result)
    print('\n')
    

