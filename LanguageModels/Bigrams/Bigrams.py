import os
import sys
import string

class Bigram:
    def __init__(self,smoothingFlag):
        self.vocabulary = {} #maps the word to the number of occurrences
        self.types = 0 #number of distinct words in corpus
        self.index_map = {}
        self.bigram_counts = []
        self.bigram_probability = []
        self.smoothingFlag = bool(smoothingFlag)

    def train(self,corpusPath):

        if self.smoothingFlag == True:
            print("Training Bigram model with smoothing... \n")
        else:
            print("Training Bigram model without smoothing... \n")

        bigrams = []
        index = 0
        with open(corpusPath,'r') as corpusFP:
            for sentence in corpusFP:
                
                tokens = self.tokenize(sentence)

                for i, token in enumerate(tokens):
                    if token in self.vocabulary:
                        self.vocabulary[token] = self.vocabulary[token] + 1
                    else:
                        self.vocabulary[token] = 1
                        self.index_map[token] = index
                        self.index_map[index] = tokens[i]
                        index = index + 1
                    
                    if i < len(tokens) - 1:
                        bigrams.append((tokens[i],tokens[i+1]))

            self.types = len(self.vocabulary)

            self.bigram_counts = [[0 for x in range(self.types)] for y in range(self.types)] 
            self.bigram_probability = [[0 for x in range(self.types)] for y in range(self.types)] 

            if self.smoothingFlag == False:

                for bigram in bigrams:
                    self.bigram_counts[self.index_map[bigram[0]]][self.index_map[bigram[1]]] = self.bigram_counts[self.index_map[bigram[0]]][self.index_map[bigram[1]]] + 1

                for row in range(self.types):
                    for col in range(self.types):
                        self.bigram_probability[row][col] = self.bigram_counts[row][col] / self.vocabulary[self.index_map[row]]
            
            elif self.smoothingFlag == True:

                for bigram in bigrams:
                    self.bigram_counts[self.index_map[bigram[0]]][self.index_map[bigram[1]]] = self.bigram_counts[self.index_map[bigram[0]]][self.index_map[bigram[1]]] + 1

                for row in range(self.types):
                    for col in range(self.types):
                        self.bigram_probability[row][col] = (self.bigram_counts[row][col] + 1) / (self.vocabulary[self.index_map[row]] + self.types)

                for row in range(0,len(self.bigram_counts)):
                    for col in range(0,len(self.bigram_counts)):
                        self.bigram_counts[row][col] = (self.bigram_counts[row][col] + 1) * (self.vocabulary[self.index_map[row]]  / (self.vocabulary[self.index_map[row]] + self.types ))
                    
        print("Bigram model training completed!\n")

        return

    def tokenize(self,sentence):
        sentence = sentence.lower() 
        sentence = sentence.replace('\n','')
        tokens = ['<s>'] + sentence.split(' ')
        for token in tokens:
            #skip punctuation and digits
            if token in string.punctuation or token in string.digits:
                tokens.remove(token)
        return tokens

    def evaluate(self,sentence):

        print("\n\t--------------------------------------------------------------------------")
        print("\t\t\t\t\tEvaluating Model")
        print("\t--------------------------------------------------------------------------")

        print('\n\n\tEvaluating Model on sentence: \"{}\"\n'.format(sentence))

        tokens = self.tokenize(sentence)

        if not self.isValidTokens(tokens):
            return

        bigrams = []
        for i in range(0,len(tokens)):
            if i < len(tokens) - 1:
                bigrams.append((tokens[i],tokens[i+1]))

        self.printBigramCounts(bigrams)

        self.printBigramProbabilities(bigrams)

        print('\tP(\"{}\") = '.format(sentence),end='')

        for idx, bigram in enumerate(bigrams):
            print('P({}|{})'.format(bigram[0],bigram[1]),end='')
            if idx < len(bigrams) - 1:
                print(' * ',end='')
        print('\n')

        probability = 1
        print('\tP(\"{}\") = '.format(sentence),end='')
        for idx, bigram in enumerate(bigrams):
            print('{}'.format(self.bigram_probability[self.index_map[bigram[0]]][self.index_map[bigram[1]]]),end='')
            probability = probability * self.bigram_probability[self.index_map[bigram[0]]][self.index_map[bigram[1]]]
            if idx < len(bigrams) - 1:
                print(' * ',end='')
        print(' = {}\n'.format(probability))
            
    def printBigramCounts(self,bigrams):

        print("\n\t**************************************")
        print("\t\t\tBigram Counts")
        print("\t**************************************\n")

        vocabulary = set()

        for bigram in bigrams:
            vocabulary.add(bigram[0])
            vocabulary.add(bigram[1])
        
        vocabulary = list(vocabulary)

        print('\t\t',end='')
        for row in vocabulary:
            print('\t{}'.format(row),end = '')
        print('\n')

        for row in vocabulary:
            print('\t{}\t\t'.format(row),end = '')
            for col in vocabulary:
                if self.smoothingFlag:
                    print('{:.3f}\t'.format(self.bigram_counts[self.index_map[row]][self.index_map[col]]),end='')
                else:
                    print('{}\t'.format(self.bigram_counts[self.index_map[row]][self.index_map[col]]),end='')
            print('\n')


    def printBigramProbabilities(self,bigrams):

        print("\n\t**************************************")
        print("\t\tBigram Probabilities")
        print("\t**************************************\n")

        vocabulary = set()

        for bigram in bigrams:
            vocabulary.add(bigram[0])
            vocabulary.add(bigram[1])
        
        vocabulary = list(vocabulary)

        print('\t\t',end='')
        for row in vocabulary:
            print('\t{}'.format(row),end = '')
        print('\n')

        for row in vocabulary:
            print('\t{}\t\t'.format(row),end = '')
            for col in vocabulary:
                print('{:.5f}\t'.format(self.bigram_probability[self.index_map[row]][self.index_map[col]]),end='')
            print('\n')

    def printProbability(self,bigram):
        try:
            word1, word2 = bigram.split(" ")
            if not self.isValidTokens([word1,word2]):
                return
            print("\nP({} | {}) = {}".format(word2,word1,self.bigram_probability[self.index_map[word1]][self.index_map[word2]]))
        except:
            print("Invalid Bigram entered!")
            
    def isValidTokens(self,tokens):
        for token in tokens:
            if token not in self.index_map:
                print("\n{} not present in Bigram Model!".format(token))
                return False
        return True

def main():
    if len(sys.argv) != 3:
        print("Expected: python3 NGrams.py <training-set> b")
        exit(1)

    corpusPath = sys.argv[1]
    smoothingFlag = int(sys.argv[2])

    if not os.path.exists(corpusPath):
        print("Invalid path entered!")  
        exit(1)

    if smoothingFlag != 0 and smoothingFlag != 1:
        print("Invalid smoothing number entered! Expected 0 or 1")
        exit(1)
    
    evaluation_sentences = ["mark antony , heere take you caesars body : you shall not come to them poet .",
    "no , sir , there are no comets seen , the heauens speede thee in thine enterprize ."]
    
    bg = Bigram(smoothingFlag)
    bg.train(corpusPath)
    

    while True:
        print("\n***************************")
        print("\tOptions Menu")
        print("***************************")
        answer = input("\nEnter: \n\t E to evaluate Bigram Model on sentences \n\t B to retrieve the probability of a Bigram \n\t C to Exit Program \n\nOption: ")
        answer = answer.lower()

        if answer == "e":
            answer = input("\nEnter: \n\t D to evaluate Bigram Model on 2 Default Sentences \n\t N to input New sentence to evaluate Bigram Model \n\nOption: ")
            answer = answer.lower()
            if answer == "d":
                print("\nEvaluating trained model...\n")
                for sentence in evaluation_sentences:
                    bg.evaluate(sentence)
            elif answer == "n":
                sentence = input("\nInput sentence: ")
                bg.evaluate(sentence)
            else:
                print("\nInvalid option entered!")
        elif answer == "b":
            bigram = input("\nInput bigram (separated by whitespace): ")
            bg.printProbability(bigram)
        elif answer == "c":
            break
        else:
            print("\nInvalid option entered!")

if __name__ == '__main__':
    main()