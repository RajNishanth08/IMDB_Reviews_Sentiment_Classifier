# IMDB Reviews Sentiment Analysis NLP
 
Objective : To classify the IMDB's Review sentiment whether it is positive or negative .

Language : Python 3.9.1

<img src="http://myfavoritewesterns.files.wordpress.com/2014/02/imdb-review.png">

Dataset Description :
Data is taken from IMDB Reviews , it contains about 50,000 reviews by various users and the dependent column is a binary class .  

Libraries : Pandas , Sklearn , NLTK , Regular Expression (re)

Requirements : 
* matplotlib==3.4.0
* nltk==3.5
* pandas==1.2.3
* scikit_learn==0.24.1
* seaborn==0.11.1

Procedure : 
 *       The dataset obtained is not a complete cleaned Dataset , so punctuations and other unwanted symbols had to be removed and everything should be converted into lowercase to avoid repeating of same words .
 *       Sentences had to be split into words and stopwords must be removed and only the important words had to be joined in a separate list . 
 *       Bag of words had to be created as it converts into vectors to train the model . Dataset has to be split into train and validation .
 *       By using MultiNomialNB and RandomForestClassifier algorithm , sentiments are predicted and performance is measured . 


Models & Accuracy : MultiNomial NB (84.4%) , RandomForestClassifier (85%)

