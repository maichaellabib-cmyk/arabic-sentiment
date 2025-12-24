# sentiment_app.py  
import streamlit as st  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB  
import re  
  
# مثال بيانات تدريبية  
data = [  
    ("هذا المنتج رائع جدًا وسأشتريه مرة أخرى.", "Positive"),  
    ("الخدمة سيئة جدًا ولن أتعامل معهم مرة أخرى.", "Negative"),  
    ("التجربة كانت عادية لا جيدة ولا سيئة.", "Neutral")  
]  
  
# Arabic preprocessing function  
def preprocess(text):  
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.strip()  
    return text  
  
texts = [preprocess(t[0]) for t in data]  
labels = [t[1] for t in data]  
  
# Vectorizer & model  
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(texts)  
model = MultinomialNB()  
model.fit(X, labels)  
  
# Streamlit interface  
st.title("تحليل المشاعر بالعربي")  
user_input = st.text_area("أدخل النص هنا:")  
  
if st.button("تحليل"):  
    input_processed = preprocess(user_input)  
    input_vec = vectorizer.transform([input_processed])  
    prediction = model.predict(input_vec)  
    st.write("المشاعر المتوقعة:", prediction[0])  
