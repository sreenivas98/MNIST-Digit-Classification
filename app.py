import gradio as gr
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import sklearn
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model

input_1 = gr.Image(shape=(28,28),image_mode='L')
 
input_2 = gr.Dropdown(["SoftMax", "KNN", "Deep Neural Network", "Decision Tree", "Random Forest"])

output = gr.Label(num_top_classes=6)

def predict_softmax(test_img):
  Softmax_model = pickle.load(open('softmax_model.pkl', 'rb'))
  predictions = Softmax_model.predict_proba(test_img)
  return {i: float(predictions[0][i]) for i in range(0,10)}
  
def predict_knn(test_img):
  Knn_model = pickle.load(open('knn_model.pkl', 'rb'))
  predictions = Knn_model.predict_proba(test_img)
  return {i: float(predictions[0][i]) for i in range(0,10)}
    
def predict_neural(test_img):
  Neural_model = load_model("deep_neural_model.h5")
  predictions = Neural_model.predict(test_img)
  return {i: float(predictions[0][i]) for i in range(0,10)}
  
def predict_tree(test_img):
  tree_model = pickle.load(open('tree_clf.pkl', 'rb'))
  predictions = tree_model.predict_proba(test_img)
  return {i: float(predictions[0][i]) for i in range(0,10)}
  
def predict_rf(test_img):
  rf_model = pickle.load(open('rf_clf.pkl', 'rb'))
  predictions = rf_model.predict_proba(test_img)
  return {i: float(predictions[0][i]) for i in range(0,10)}

def predictDigitClass(test_img,chosen_model):
  test_img_flatten=test_img.reshape(-1,28*28)
  if chosen_model == "SoftMax":
    fashionProbs = predict_softmax(test_img_flatten)
    return fashionProbs
  elif chosen_model == "KNN":
    fashionProbs = predict_knn(test_img_flatten)
    return fashionProbs  
  elif chosen_model == "Deep Neural Network":
    fashionProbs = predict_neural(test_img_flatten)
    return fashionProbs 
  elif chosen_model == "SVM":
    fashionProbs = predict_svm(test_img_flatten)
    return fashionProbs     
  elif chosen_model == "Decision Tree":
    fashionProbs = predict_tree(test_img_flatten)
    return fashionProbs
  elif chosen_model == "Random Forest":
    fashionProbs = predict_rf(test_img_flatten)
    return fashionProbs
    
gr.Interface(fn=predictDigitClass,inputs=[input_1,input_2],outputs=output).launch(debug=True)
