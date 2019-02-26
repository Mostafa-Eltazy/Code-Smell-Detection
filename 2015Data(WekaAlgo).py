#this is for loading the dataset
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random


import weka.core.jvm as jvm
jvm.start()
#Loding the GOD CLASS data set --------------------
loader = Loader(classname="weka.core.converters.ArffLoader")
God_Class_data = loader.load_file("./Evaluationdataset/god-class.arff")
God_Class_data.class_is_last() # To make thr classfier class the last attribute 
print(God_Class_data)
print("--------------God class classfication Algorithms results-------------------------")
#This is for classfication using [trees.Random Forest]---------------------------------------------
print("-------------------The results of Random Forest-------------------------")
classifier_secondealgorithm = Classifier(classname="weka.classifiers.trees.RandomForest")
evaluation = Evaluation(God_Class_data)                     # initialize with priors
evaluation.crossvalidate_model(classifier_secondealgorithm, God_Class_data, 10, Random(42))  # 10-fold CV
print(evaluation.summary())
print("pctCorrect: " + str(evaluation.percent_correct))
print("incorrect: " + str(evaluation.incorrect))




