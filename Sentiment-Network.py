import time
import sys
import numpy as np
from collections import Counter

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, min_count, polarity_cutoff, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        self.min_count = min_count 
        self.polarity_cutoff = polarity_cutoff
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):

        positive_counts = Counter()
        negative_counts = Counter() 
        total_counts = Counter() 
        for i in range(len(reviews)) :
            for w in reviews[i].split() :
                total_counts[w] += 1
                if labels[i] == 'POSITIVE' :
                    positive_counts[w] += 1
                else :
                    negative_counts[w] += 1
            
        review_vocab = set()
        for term,cnt in list(total_counts.most_common()):
            if(cnt > self.min_count and positive_counts[term] != 0):
                pos_neg_ratio = np.log(positive_counts[term] / float(negative_counts[term]+1))
                if pos_neg_ratio < -1*self.polarity_cutoff or pos_neg_ratio > self.polarity_cutoff  :
                    review_vocab.add(term)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set([l for l in labels])
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i in range(self.review_vocab_size) :
            self.word2index[self.review_vocab[i]] = i
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i in range(self.label_vocab_size) :
            self.label2index[self.label_vocab[i]] = i
         
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))  
        
        self.layer_1 = np.zeros((1,self.hidden_nodes))
    
        
                
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        return 1. / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        return output*(1 - output)

    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = []
        for r in training_reviews_raw :
            tr = set()
            for w in r.split(" ") :
                if( w in self.word2index.keys()) :
                    tr.add(self.word2index[w])
            training_reviews.append(list(tr))
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            review, label = training_reviews[i], training_labels[i]

            #forward_pass
            self.layer_1 *= 0
            for idx in review :
                self.layer_1 += self.weights_0_1[idx]
            output_h = self.sigmoid(np.dot(self.layer_1,self.weights_1_2))

            #back propogation            
            error =  output_h - self.get_target_for_label(label)
            grad = self.sigmoid_output_2_derivative(output_h)
            error_term_2 = error*grad
            error_term_1 = np.dot(error_term_2 , self.weights_1_2.T)

            self.weights_1_2 -= np.dot(self.layer_1.T, error_term_2)*self.learning_rate

            for idx in review :
                self.weights_0_1[idx] -= error_term_1[0]*self.learning_rate
            # Keep track of correct predictions. To determine if the prediction was
            #      correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if -0.5 < error and error < 0.5 :
                correct_so_far += 1
                
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review_raw):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        review = set() 
        for w in review_raw.split(" ") :
            if w in self.word2index.keys() :
                review.add(self.word2index[w])
        review = list(review)

        self.layer_1 *= 0 
        for idx in review :
            self.layer_1 += self.weights_0_1[idx]
        output_h = self.sigmoid(np.dot(self.layer_1,self.weights_1_2))
        # The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        if output_h >= 0.5 :
            return 'POSITIVE'
        return 'NEGATIVE'

if __name__ == "__main__":
g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))    
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close() 
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.09,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])


'''
Training
Progress:99.9% Speed(reviews/sec):1361. #Correct:20553 #Trained:24000 Training Accuracy:85.6%

Testing
Progress:99.9% Speed(reviews/sec):1759. #Correct:856 #Tested:1000 Testing Accuracy:85.6%

'''

#Speed(reviews/sec):530.5 
#Correct:20092 
#Trained:24000 
#Training Accuracy:83.7%

'''Testing'''
#Speed(reviews/sec):1081. 
#Correct:847 
#Tested:1000 
#Testing Accuracy:84.7%