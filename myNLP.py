#Natural Language Processing

#Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Import dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#Cleaning the text

"""import re
import nltk
review=re.sub('[^a-zA-Z]',' ',dataset['Review'][0]) #removing non alphabet letters
review=review.lower()  #changing to lowercase
nltk.download('stopwords')  #download list of words which are irrelevant
review=review.split() #converting review to list
from nltk.corpus import stopwords
#removing irrelevant characters like this that prepositions hmm etc...
#review=[word for word in review if not word in set(stopwords.words('english'))]
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer() #we want to keep only root word for ex loved will be replace by love
review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review=' '.join(review) #reconverting review list to string """
#whole above code for whole dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split() 
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
# Creating Bag of Words Model
#it will create a table rows corresponding to 1000 reviews and columns to each word
#it will contain how many times a word (column) has occured in each row
#it will contain a lot of zeroes therefore a sparse matrix 
    
    

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #max_feautres will include only most frequently used 1500 columns
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
        
        # Naive Bayes
"""
#Applying Naive Bayes Classification Model to classify whether a review is +ve or -ve

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# cm shows accuracy is 73 %

"""
 
   #Decision Tree
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy = 66%
#precision = 80%
#Recall = 61%
#F1 Score = 69%
    