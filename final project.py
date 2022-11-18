import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from wordcloud import WordCloud,STOPWORDS
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from PIL import Image
#loading screen
st.title('SPAM SMS COLLECTION')
dataset_loc="G:\pc backup\spyder\SMSSpamCollection.txt"
image_loc = Image.open("G:\\pc backup\\spyder\\image.png")
df=pd.read_csv(dataset_loc,sep="\t",names=['target','message'])

#sidebar
st.sidebar.subheader("SPAM DETECTOR SYSTEM")
xz=st.sidebar.selectbox('WHAT ARE YOU LOOKING FOR?',('Information','Data Preview','Word-Cloud Wall','Count Plot','Bag of Words','Logistic Regression','Decision Tree','Support Vector Machine'))


def preprocess(abc):
    #preprocessing
    lemmatizer=WordNetLemmatizer()
    letters_only=re.sub("[^a-zA-Z]"," ",abc) #removing special symbols
    lower_words=letters_only.lower() #converting to the lowercase to all
    words=lower_words.split() #tokenizing
    words=[word for word in words if not word in stopwords.words("english")] #removing stop words
    words = [lemmatizer.lemmatize(word) for word in words] #lemmatization
    clean_sentence=" ".join(words)
    return clean_sentence

def get_sidebar(xz):
    #bag of words
    df["clean_text"]=df["message"].apply( lambda x : preprocess(x))
    #st.subheader('performing train-test-split')
    training_data,testing_data=train_test_split(df,train_size=0.73,random_state=42) #Train,Test,Split
    testing_target=testing_data["target"]
    r=[]
    for word in training_data["clean_text"]:
        r.append(word)
        s=[]
    for word in testing_data["clean_text"]:
        s.append(word)   
      
    vectorizer    = CountVectorizer(analyzer = "word")
    training_text = vectorizer.fit_transform(r)
    testing_text  = vectorizer.transform(s) 
    training_text = training_text.toarray()
    testing_text  = testing_text.toarray()
    if xz == 'Information':
        st.image(image_loc, use_column_width = True, channels="BGR")
        st.subheader('SOURCE: ')
        st.code('''            Tiago  Almeida (talmeida ufscar.br) 
            Department of Computer Science 
            Federal University of Sao Carlos (UFSCar) 
            Sorocaba, Sao Paulo - Brazil 
    
            JosÃ© MarÃ­a GÃ³mez Hidalgo (jmgomezh yahoo.es) 
            R&D Department Optenet 
            Las Rozas, Madrid - Spain''')
        st.write('\n')    
        st.subheader('DATASET INFORMATION: ')
        st.code('''            This corpus has been collected from free or free for research sources at the Internet: 

            -> A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. 
            -> A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. 
            -> A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis available at [Web Link]. 
            -> Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages and it is public available at: [Web Link]. This corpus has been used in the following academic researches: 

            [1] GÃ³mez Hidalgo, J.M., Cajigas Bringas, G., Puertas Sanz, E., Carrero GarcÃ­a, F. Content Based SMS Spam Filtering. Proceedings of the 2006 ACM Symposium on Document Engineering (ACM DOCENG'06), Amsterdam, The Netherlands, 10-13, 2006. 
                                                                                                                                                                                                  
            [2] Cormack, G. V., GÃ³mez Hidalgo, J. M., and Puertas SÃ¡nz, E. Feature engineering for mobile (SMS) spam filtering. Proceedings of the 30th Annual international ACM Conference on Research and Development in information Retrieval (ACM SIGIR'07), New York, NY, 871-872, 2007. 

            [3] Cormack, G. V., GÃ³mez Hidalgo, J. M., and Puertas SÃ¡nz, E. Spam filtering for short messages. Proceedings of the 16th ACM Conference on Information and Knowledge Management (ACM CIKM'07). Lisbon, Portugal, 313-320, 2007.''')    
        st.write('\n')
        st.subheader('ATTRIBUTE INFORMATION: ')
        st.code('''            The collection is composed by just one text file, where each line has the correct class followed by the raw message. We offer some examples bellow: 

                ham What you doing?how are you? 
                ham Ok lar... Joking wif u oni... 
                ham dun say so early hor... U c already then say... 
                ham MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H* 
                ham Siva is in hostel aha:-. 
                ham Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor. 
                spam FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop 
                spam Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B 
                spam URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU 

            Note: the messages are not chronologically sorted''')
        st.write('\n')
        st.subheader('RELEVANT PAPERS: ')
        st.code('''            We offer a comprehensive study of this corpus in the following paper. This work presents a number of statistics, studies and baseline results for several machine learning methods. 

            Almeida, T.A., GÃ³mez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011. ''')
        st.write('\n')                                                                                                                                                                                            
    elif xz == 'Data Preview':
        # Preview of the dataset
        st.header("DATA PREVIEW")
        preview = st.radio("Choose Head/Tail?", ("Top", "Bottom"))
        if(preview == "Top"):
            st.write(df.head())
        if(preview == "Bottom"):
            st.write(df.tail())
        # display the whole dataset
        if(st.checkbox("Show complete Dataset")):
            st.write(df)
        # Show shape
        if(st.checkbox("Display the shape")):
            st.write(df.shape)
            dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
            if(dim == "Rows"):
                st.write("Number of Rows", df.shape[0])
            if(dim == "Columns"):
                st.write("Number of Columns", df.shape[1])
        # show columns
        if(st.checkbox("Show the Columns")):
            st.write(df.columns)
    elif xz == 'Word-Cloud Wall':
        #wordcloud
        st.header("WORD-CLOUDS WALL")
        type = st.radio("Choose the sentiment?", ("ham", "spam"))
        if(type=='spam'):
            temp_df = df.loc[df.target=='spam', :]
            words = ' '.join(temp_df['message'])
            cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
            wc = WordCloud(stopwords=STOPWORDS, background_color='black', width=1600, height=1200).generate(cleaned_word)
            wc.to_file("G:\pc backup\spyder\wc.png")
        else:
            temp_df = df.loc[df.target=='ham', :]
            words = ' '.join(temp_df['message'])
            cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
            wc = WordCloud(stopwords=STOPWORDS, background_color='black', width=1600, height=1200).generate(cleaned_word)
            wc.to_file("G:\pc backup\spyder\wc.png")    
        st.image(Image.open("G:\pc backup\spyder\wc.png"), use_column_width = True, channels="BGR")
    elif xz == 'Count Plot':
        #countplot
        st.header('COUNT PLOT')
        airline = st.radio("Choose an message type", ('ham','spam'))
        temp_df = df.loc[df['target']==airline, :]
        st.write(sns.countplot(x='target', order=['ham','spam'], data=temp_df))
        st.pyplot()
    #bag of words output
    elif xz == 'Bag of Words':
        st.header('RESULTS OF BAG-OF-WORDS')
        st.write("Total words:", len(vectorizer.vocabulary_))
        st.write("Shape of training text", training_text.shape)
    #logistic regression
    elif xz == 'Logistic Regression':
        st.header('RESULTS OF LOGISTIC REGRESSION')
        lr=LogisticRegression()
        lr.fit(training_text,training_data["target"])
        lr_prediction= lr.predict(testing_text)
        lr_accuracy  = metrics.accuracy_score(testing_target,lr_prediction)
        lr_matrix    = metrics.confusion_matrix(testing_target,lr_prediction) 
        st.write('- Accuracy of logistic regression')
        st.write(' ',lr_accuracy)
        st.write('\n- Confusion matrix of data')
        st.write(' ',lr_matrix)
        st.write('\n- Classification report of logistic regression')
        st.text(metrics.classification_report(testing_target,lr_prediction))
        st.write('\n- Heat-map of data')
        (sns.heatmap(lr_matrix,annot=True))
        st.pyplot()
        #spamdetect
        st.subheader('MESSAGE PREDICTION USING LOGISTIC REGRESSION')
        q=st.text_input('ENTER THE MESSAGE','type here..')
        if st.button('submit'):
            st.text(q)
            a= lr.predict(vectorizer.transform([preprocess(q)]).toarray())
            if(a=="spam"):
                st.write("SPAM SMS!")
            else:
                st.write("IMPORTANT SMS!")
            
    #decision tree classifier
    elif xz == 'Decision Tree':
        st.header('RESULTS OF DECISION TREE CLASSIFIER')
        dt=DecisionTreeClassifier()
        dt.fit(training_text,training_data["target"])
        dt_prediction= dt.predict(testing_text)
        dt_accuracy  = metrics.accuracy_score(testing_target,dt_prediction)
        dt_matrix    = metrics.confusion_matrix(testing_target,dt_prediction)
        st.write('- Accuracy of decision tree classifier')
        st.write(dt_accuracy)
        st.write('\n- Confusion matrix of data')
        st.write(dt_matrix)
        st.write('\n- Classification report of decision tree classifier')
        st.text(metrics.classification_report(testing_target,dt_prediction))
        st.write('\n- Heat-map of data')
        sns.heatmap(dt_matrix,annot=True)
        st.pyplot()
        #spamdetect
        st.subheader('MESSAGE PREDICTION USING DECISION TREE CLASSIFIER')
        w=st.text_input('ENTER THE MESSAGE','type here..')
        if st.button('submit'):
            st.text(w)
            e= dt.predict(vectorizer.transform([preprocess(w)]).toarray())
            if(e=="spam"):
                st.write("SPAM SMS!")
            else:
                st.write("IMPORTANT SMS!")
    #suport vector machine
    else:
        st.header('RESULTS OF SUPPORT VECTOR MACHINE (SVM)')
        svm=SVC()
        svm.fit(training_text,training_data["target"])
        svm_prediction= svm.predict(testing_text)
        svm_accuracy  = metrics.accuracy_score(testing_target,svm_prediction)
        svm_matrix    = metrics.confusion_matrix(testing_target,svm_prediction)
        st.write('- Accuracy of support vector machine')
        st.write(svm_accuracy)
        st.write('\n- Confusion matrix of data')
        st.write(svm_matrix)
        st.write('\n- Classification report of support vector machine')
        st.text(metrics.classification_report(testing_target,svm_prediction))
        st.write('\n- Heat-map of data')
        sns.heatmap(svm_matrix,annot=True)
        st.pyplot()
        #spamdetect
        st.subheader('MESSAGE PREDICTION USING SUPPORT VECTOR MACHINE')
        r=st.text_input('ENTER THE MESSAGE','type here..')
        if st.button('submit'):
            st.text(r)
            t= svm.predict(vectorizer.transform([preprocess(r)]).toarray())
            if(t=="spam"):
                st.write("SPAM SMS!")
            else:
                st.write("IMPORTANT SMS!")
get_sidebar(xz)   
    
       