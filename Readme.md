# Sentiment-Analysis in python From Scratch (Sentiment-Network.py)
  
### 1. Build Neural Network.
![Network Image](https://github.com/dragod812/Sentiment-Analysis/blob/master/sentiment_network_2.png)<br>
You can specify the number of input_nodes, hidden_nodes and output_nodes. The input is just a<br> vector of all the word counts that are in the review and a label for each review ('POSITIVE' <br>or 'NEGATIVE'). Gave an accuracy of about 60%.
### 2. Reduce Noice in the Data.
We decided to reduce the noise in the data by taking a boolean value for the words rather than a count. This way we got increased accuracy. Accuracy of nearly 80%.
### 3. Make Network more efficient.
The input vector for this neural network is definitely sparse therefore we are doing a lot of wasted computation. To avoid this we added the weights from the input to the hidden layer only if the weights source node word is present in the review. we also back propogate only those weights that are present in the input.
### 4. Reducing Noise by Strategically Reducing the Vocabulary.
The title speaks for itself. We have a Count threshold on the words and create a polarity metric to measure whether a word is 'POSITIVE' or 'NEGATIVE' and remove the words that lie somewhere in the middle.

```
Training
Progress:99.9% Speed(reviews/sec):1361. #Correct:20553 #Trained:24000 Training Accuracy:85.6%

Testing
Progress:99.9% Speed(reviews/sec):1759. #Correct:856 #Tested:1000 Testing Accuracy:85.6%
```
