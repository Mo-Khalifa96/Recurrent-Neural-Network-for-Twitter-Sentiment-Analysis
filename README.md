# Recurrent Neural Network for Twitter Sentiment Analysis
<br>

## About The Project 
**This project involves the implementation and employment of a recurrent neural network with embedding and advanced attentional mechanisms for sentiment analysis, particularly training on and analyzing tweets. Sentiment analysis, as implied by the term, involves analyzing the content of a given text to determine the emotional undertone or sentiment most appropriate to it. This practice is particularly helpful for a variety of purposes, chief among them is understanding and learning from public perception and public opinion, whether to understand the public stance towards a particular topic, company, or service; for research and development; to better manage and monitor social spaces (for instance by uprooting hateful, inciting or inflammatory content); or even to better understand the general social and political climate, which has a direct bearing on the stocks and forex markets and can thus inform decisions related thereto. Indeed, with the unrelenting increased digitization of the public sphere, sentiment analysis, especially as applied to large-scale social media platforms like X/twitter, has become pivotal for companies everywhere to better manage the public reception of their products and services and to ensure their betterment and meet customers' expectations aptly, thus greatly aiding companies' pursuit of growth and quality. To that end, I endeavoured to develop a recurrent neural network with various attentional and optimization mechanisms and train it on a large dataset of tweets to learn to distinguish between different sentiment and identify the correct sentiment for each individual tweet.** <br>
<br>
**Prior to model development, the dataset was quickly inspected and cleaned before being thoroughly preprocessed and prepared for training. Preparatory steps included removing hyperlinks, hashtags, stop words, emojis and emoticons, and lemmatization. The tweets were then tokenized and padded to be fed appropriately to the model. A recurrent neural network was then developed and trained for sentiment analysis. It was also endowed with several attentional capabilities to facilitate sentiment analysis. As such, this network roughly consisted of the following: an embedding layer for word embedding; a mask layer, which specialized in masking sentiment or emotional words, adding more emphasis to them during training; a bidirectional Long-Short Term Memory (LSTM) layer to learn context and semantic dependencies in the data; a self-attention for added importance on the most relevant parts of the text; and finally a dense layer with 3 units for classification. Each layer fedforward to the next one before terminating at the classification layer to identify the sentiment appropriate to a given tweet. Finally, the network was then tested on a separate testing set for a final evaluation. The network yielded considerably favorable results.** <br>

<br>

**Overall, the project is comprised of 3 sections: <br>
&emsp; 1) Reading and Inspecting the Data <br>
&emsp; 2) Data Preparation and Preprocessing <br> 
&emsp; 3) Model Development and Evaluation** <br>

<br>
<br>

## About The Data  
**The dataset presented here was taken from Kaggle, which you can quickly access from the following [link](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset/). This dataset is comprised of approximately 27,500 tweets, each tweet labelled in advance with the sentiment appropriate to it. Sentiment labels present are simply "positive", "negative", or "neutral". Each row in the data includes the tweet's text content and a corresponding sentiment label. The goal here, as mentioned, is to develop a neural network that learns from this dataset to correctly identify the sentiment or emotion that perfectly fits a given post.**<br> 
<br>

**You can view each column and its description in the table below:** <br><br>  

| **Variable**      | **Description**                                                                                         |
| :-----------------| :------------------------------------------------------------------------------------------------------ |
| **textID**       | Unique identifier for each tweet                                                             |
| **text**| Raw text of the tweet                                                     |
| **selected_text**    | Most relevant or informative part of the tweet for deciding sentiment                                                                                       |
| **sentiment**  | Sentiment label corresponding to the tweet (neutral, positive, or negative)              |

<br>
<br>


## Quick Access 
**For quick access to preview the project, I provided two links, both of which will direct you to a Jupyter Notebook with all the code and corresponding output, supplied with observations and explanations along the way. The first link allows you to view the project only, the second link allows you to view the project as well as interact with it directly and reproduce the results if you prefer so. To execute the code, please make sure to run the first two cells first in order to install and import the Python packages necessary for the task. To run any given block of code, simply select the cell and click on the 'Run' icon on the notebook toolbar.**
<br>
<br>

***To view the project only, click on the following link:*** <br>
https://nbviewer.org/...
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/...
<br>
<br>


