# ES654-assignment-3-akcgjc007

## [Document](https://docs.google.com/document/d/1i3NWa2aViJuv8R1qAAC83mv1lXlX6gKA5QeU2IJ2uJY/edit)

## Problem statement
Assignment 3
Total Marks: 20
Deadline: April 18, 11:59 PM

Create your own repo and write code in a similar fashion as you did for earlier assignments. We are not providing any sample code this time. Any commits after the deadline will not be acceptable. 

If you have any questions, please ask them on MS Teams below this post. Please copy Mrinal and Nidhin on Teams. 

 - Implement Unregularised Logistic regression for 2-class problem using:
   - Update rules mentioned in class slides with gradient descent [Slide 26 from https://nipunbatra.github.io/ml2021/lectures/logistic-regression.pdf] [1 mark]
   - Use Jax/Autograd to automatically compute gradient and solve with gradient descent [1 mark]
   - Using breast cancer dataset and K=3 folds present the overall accuracy. [1 mark]
   - Plot decision boundary for 2d input data where you can choose any two pairs of features to visualise [1 mark]
 - Implement L1 and L2 regularised logistic regression for 2-class problem using:
   - Jax/Autograd [1 mark]
   - Using nested cross-validation find the optimum lambda penalty term for L2 and L1 regularisation. From the L1 regularisation, can you infer the “more” important features? [2 marks]
 - Implement K-class Logistic regression using:
   - Update rules in slides [Slide 38 from https://nipunbatra.github.io/ml2021/lectures/logistic-regression.pdf] [1 mark]
   - Jax/Autograd [1 mark]
   - Using Digits dataset and stratified (K=4 folds) visualise the confusion matrix and present overall accuracy. Which two digits get the most confused? Which is the easiest digit to predict?  [3 mark]
   - Use PCA (as blackbox) from sklearn and project the digit data to 2 dimensions and make a scatter plot with different colours showing the different digits. What can you infer from this plot? [1 mark]
 - What is time and space complexity for Logistic regression learning and prediction? [1 mark]
 - Create a fully connected NN (MLP) where the input is X, y, [n1, n2, …, nh] where ni is the number of neurons in i^th hidden layer, [g1, g2, …, gh] where gi in {‘relu’, ‘identity’, ‘sigmoid’} are the activations for i^th layer. You should use Jax for backpropagation. You should write the forward pass yourself. [3 marks]
 - Test NN code for simple classification (Digits dataset)  and regression dataset (Boston housing) both using 3-fold CV. You can choose the number of layers and activations of your choice. [3 marks]

## Datasets:
 - Digits dataset https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
 - Breast cancer dataset https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
 - Boston housing dataset https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston





