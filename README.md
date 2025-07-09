# NLP project



\# 📊 Sentiment Analysis on Restaurant Reviews



This project performs a full machine learning pipeline to classify restaurant reviews as \*\*positive or negative\*\*.  

It uses natural language processing techniques, from named entity recognition (NER) to Bag of Words, and evaluates multiple models.



---



\## 🚀 Project Overview



The main steps in this project include:



1\. \*\*Data Preprocessing:\*\*

&nbsp;  - Load TSV file with restaurant reviews and labels (positive/negative).

&nbsp;  - Remove punctuation from all reviews.

&nbsp;  - Lemmatize words to reduce them to their basic form (using spaCy).



2\. \*\*Feature Engineering:\*\*

&nbsp;  - Create two types of text representations:

&nbsp;    - \*\*Entity-based features:\*\* Extracted with NER (Named Entity Recognition) using spaCy.

&nbsp;    - \*\*Bag of Words (BoW):\*\* Using `CountVectorizer` on the lemmatized reviews.



3\. \*\*Model Training and Evaluation:\*\*

&nbsp;  - Train and evaluate five models on each representation:

&nbsp;    - Logistic Regression

&nbsp;    - Random Forest Classifier

&nbsp;    - Support Vector Machine (SVC)

&nbsp;    - Multinomial Naive Bayes

&nbsp;    - K-Nearest Neighbors (KNN)

&nbsp;  - Metrics include:

&nbsp;    - Accuracy

&nbsp;    - Precision

&nbsp;    - Recall

&nbsp;    - F1 Score

&nbsp;    - Confusion Matrix



4\. \*\*Model Selection:\*\*

&nbsp;  - Based on evaluation, Random Forest was selected as the best performing model.



---



\## 🔍 Results Summary



| Model               | Accuracy | Precision | Recall | F1 Score |

|----------------------|----------|-----------|--------|----------|

| Logistic Regression  | 0.76     | 0.81      | 0.70   | 0.75     |

| Random Forest        | 0.78     | 0.80      | 0.75   | 0.78     |

| SVM                  | 0.78     | 0.81      | 0.74   | 0.77     |

| Multinomial NB       | 0.77     | 0.81      | 0.71   | 0.76     |

| KNN                  | 0.66     | 0.66      | 0.73   | 0.69     |



✅ \*\*Chosen Model:\*\* Random Forest Classifier — achieved best balance of precision, recall, and F1 score.



---



\## 🛠 Requirements



Install the required Python packages:



```bash

pip install pandas numpy spacy scikit-learn tqdm

python -m spacy download en\_core\_web\_sm





