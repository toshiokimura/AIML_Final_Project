# Final-Project

## Summary

### By creating the self-learning neural network model from scratch and applying the assumption that multiple sequential images represent the same object, the model successfully improves prediction accuracy on new datasets on the fly without any external correct data label feedback, which is beneficial for efficient model refinement over time.

<BR>

## 1. Purpose
### Updating AI/ML model on the fly

#### This project aims to update a neural network AI/ML model on the fly using input data during its use stage, with a focus on image recognition.

The typical way to define a model is to first apply training and test data to create multiple possible models. Then, the models are evaluated based on metrics like accuracy, loss, and load, etc. Finally, the model with the best set of parameters is chosen. Once the model is finalized, it can be used in an application, however, there is no way to improve it after deployment.

The reason why we may want to improve the model after deployment is that data characteristics can change over time. For example, the shapes of numeral digits or vehicles may evolve gradually, which causes model deterioration. The challenge of adjusting the model during its use stage lies in determining how to obtain the correct labels for new data when the model cannot predict with 100% confidence.

One approach to address this issue is to consider all predictions from the current model as correct. This can be reasonable, especially when the model is well-trained and delivers over 90% accuracy. The model can be updated based on new data and the predicted labels as a compromise.

Another method is to use time sequencing. For example, while we might not correctly predict a small, distant object at first, we may identify it as a car once it gets closer and the image becomes clearer. In this case, we can retroactively learn that the previous small object was also a car. This new data, with its presumably correct predicted label, can then be used for additional training.

For this project, to simplify the process, I will use handwritten digit data (0-9) instead of car images to improve the model in both scenarios: (i) shape changes over time and (ii) object size changes over short periods.

<p align = "center">
  <img src="https://github.com/user-attachments/assets/0d66cbd0-54e5-449f-9e11-8ca522b755d4" style="width: 50%; height: 50%;" />
  
<BR>
<BR>

## 2. Data Preparation

### Preparing Training and Test Data

#### Download the numeral digit dataset and create new datasets that imitate older or distant object data.

This project uses the MNIST dataset for handwritten digit samples. The MNIST dataset includes 70,000 images in total, with 60,000 examples in the training set and 10,000 examples in the testing set. Both sets contain labeled images of 10 digits (0 to 9) in a 28x28 pixel grayscale format. Four new data types are created from the original dataset: the 1/2 size and 3/4 size datasets imitate older or more distant objects, while center-aligned and top-left-aligned data are used to explore whether recognition accuracy depends on the object's location.
 
- Original (Center, 28x28)	: Center-aligned, Original size (Original data)
- 21c (Center, 21x21)		: Center-aligned, 3/4 size (Old object data, Distant object data)
- 14c (Center, 14x14)		: Center-aligned, 1/2 size (Older object data, Farther distant object data)
- 21tl (Top Left, 21x21)		: Top-left corner aligned, 3/4 size (Old object data, Distant object data)  
- 14tl (Top Left, 14x14)		: Top-left corner aligned, 1/2 size (Older object data, Farther distant object data)

As shown below, the new datasets were created as intended.

<p align = "center">
  <img src="https://github.com/user-attachments/assets/ca0a4c35-db2d-43f0-b3c0-7034390c5e90" style="width: 75%; height: 75%;" />

<p align = "center">
  <img src="https://github.com/user-attachments/assets/42f8dfdb-8f30-46a8-90cb-54b1f48f296f" style="width: 50%; height: 50%;" />

<BR>
<BR>

## 3. Model Generation

### Constructing a Neural Network Model

#### Create a neural network model from scratch with a 1- or 2-hidden-layer architecture.

One of the challenges of using TensorFlow Keras-based models is their lack of flexibility. In this project, the model parameters, Wx and bx, need to be updated based on the input data during the model's use stage, which is not possible with the default TensorFlow Keras models. Therefore, I decided to create the model from scratch. The code linked below was used as a reference:


    Simple MNIST NN from scratch (numpy, no TF/Keras)
    https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras

Based on this model, new features have been implemented, as highlighted below.

| Feature            | Reference project     | This project                                                                                                     |
|--------------------|-----------------------|------------------------------------------------------------------------------------------------------------------|
| Hidden layer       | One hidden layer only | One hidden layer<br>Two hidden layer                                                                             |
| Hidden layer width | Fixed to 10           | Flexible width options                                                                                           |
| Output width       | 10                    | 10                                                                                                               |
| Batch size         | All data only         | Flexible batch size options                                                                                      |
| Epoch              | Available             | Available                                                                                                        |
| Alpha              | Available             | Available                                                                                                        |
| Predict and Learn  | -                     | Available <br>Wx, bx available from the pre-trained model<br>Wx, bx can be updated in the actual model use stage |
| Graph              | -                     | Graph for accuracy history through model generation                                                              |


<br>
<br>

<img src="https://github.com/user-attachments/assets/e7e184ed-224d-49ae-b84e-501122ebcf49" style="width: 75%; height: 75%;" />


<BR>
<BR>

## 4. Model Application

### Applying Nural Network Model

#### Apply the neural network model with different parameter settings and monitor accuracy.

<br>

### (1)	Standard Model Evaluation

Several standard models are evaluated to decide which model to use for the on-the-fly learning.

<BR>

**Standard Model Test Results (Model 1 - 11)**

| Model                           | 1        | 2                 | 3        | 4        | 5        | 6     | 7     | 8     | 9     | 10                  | 11                                 |
|---------------------------------|----------|-------------------|----------|----------|----------|-------|-------|-------|-------|---------------------|------------------------------------|
| Target data (train and test)    | Original | Original          | Original | Original | Original | 21c   | 14c   | 21tl  | 14tl  | Original + 21c +14c | Original + 21c +14c + 21tl + 14tl  |
| Hidden Layer                    | 1        | 1                 | 2        | 1        | 1        | 1     | 1     | 1     | 1     | 1                   | 1                                  |
| Width Layer 1                   | 10       | 20                | 10       | 10       | 10       | 10    | 10    | 10    | 10    | 10                  | 10                                 |
| Width Layer 2                   | [10]     | [10]              | 10       | [10]     | [10]     | [10]  | [10]  | [10]  | [10]  | [10]                | [10]                               |
| Width Layer 3                   | -        | -                 | [10]     | -        | -        | -     | -     | -     | -     | -                   | -                                  |
| batch                           | 100      | 100               | 100      | 2        | 6000     | 100   | 100   | 100   | 100   | 100                 | 100                                |
| alpha                           | 0.1      | 0.1               | 0.1      | 0.1      | 0.1      | 0.1   | 0.1   | 0.1   | 0.1   | 0.1                 | 0.1                                |
| epoch                           | 60       | 60                | 60       | 60       | 300      | 60    | 60    | 60    | 60    | 60 (20x3)           | 60 (12x5)                          |
| Predict and Learn               | 0        | 0                 | 0        | 0        | 0        | 0     | 0     | 0     | 0     | 0                   | 0                                  |
| Accuracy for the target dataset | 93.7%    | 95.9%             | 93.7%    | 87.8%    | 91.2%    | 93.6% | 92.7% | 93.8% | 93.1% | 90.2%, 90.9%, 88.8% | 86.0%, 86.6%, 86.4%, 83.0%, 86.2%  |
| Processing time (s)             | 51.6     | 71.7              | 64.8     | 168.0    | 253.2    | 57.8  | 54.7  | 61.0  | 54.2  | 28.2                | 42.6                               | 
| Comments                        | OK       | Improved a little | OK       | Slow     | Slow     | OK    | OK    | OK    | OK    | OK                  | OK                                 |

<BR>

The table above shows the overall results, with parameter changes highlighted in blue. Based on the results, Model 1 with basic parameter settings performs well. Other key findings include:

   -	Increasing the width from 10 to 20 or adding a second hidden layer does not significantly improve accuracy.

   -	A batch size of 100 processes training faster compared to a batch size of 2 or 6,000.

   -	The model with the original dataset (Model 1) achieves the second-best accuracy among the others (Model 6 – Model 9), and these newly created datasets (Model 6 – Model 9) show relatively good accuracy.

   -	The dataset that combines multiple datasets does not significantly degrade accuracy (Model 10 and Model 11).

   -	Datasets with a similar size or location to the trained dataset are well recognized.

   -	The final W1 has different shapes of weight intensities.

Based on these findings, the W1 and b1 generated from Model 1 will be applied in the next step for the on-the-fly learning evaluation.

<BR>



Accuracy Improvements Over Epoch (Model 1)
-	The best accuracy is achieved with the original data (28c), followed by 21c, 14c, 21tl, and 14tl, in that order.
![image](https://github.com/user-attachments/assets/16ec69e6-1ad6-48eb-a00a-782b259eb126)

<BR>

The evolution of W1 in Model 1 (Epoch 0, Epoch 5, and Epoch 60):
-	The W1 at Epoch 0 (top) is random. By Epoch 5 (middle), W1 is already beginning to show the direction it will take by Epoch 60 (bottom).
![image](https://github.com/user-attachments/assets/c6b6d653-2d9f-41a5-9c8d-ea967b9474b5)
![image](https://github.com/user-attachments/assets/9b16887a-a893-4a3f-96f8-100a72323eb5)
![image](https://github.com/user-attachments/assets/4235210c-9b6e-4ed6-8caf-982ac29de1a0)

<BR>

The evolution of W1 in Model 4 (Epoch 60): Batch size 2
-	Outside the effective center area, the weights are well averaged.
![image](https://github.com/user-attachments/assets/e6014aec-71c5-4fa9-b05a-1ab89e7f62fa)

<BR>

The evolution of W1 in Model 9 (Epoch 60): 14tl dataset
-	The shapes are observed only in the top left corner.
![image](https://github.com/user-attachments/assets/6af7002e-67ab-4e2c-9efc-f7f2b603d364)

<BR>


### (2) On-the-Fly Learning Model Evaluation

For the on-the-fly learning model evaluation, W1, b1, W2, and b2 from the training results of Model 1 are basiucally used as default parameters. Model 1, with these parameters, is well-trained on the original data. There are three types of on-the-fly learning:

<ol type="a">
  <li> Supervised Learning: This method receives correct data labels, similar to a teacher providing the correct label for each new data point. The model is updated based on the correct data.</li>
  <li> Unsupervised Learning (1): This method does not receive correct data labels. Instead, the model uses predicted labels to update itself on-the-fly to adapt to new data. The results are gradually reflected in the model’s parameters, either one by one or in small batches</li>
  <li> Unsupervised Learning (2): This method is essentially the same as Unsupervised Learning (1), but the model uses one predicted label based on eacch of the three data points from a single pair.</li>
</ol>

<BR>

**On-the-Fly Learning Model Test Results (Model A1 - C1)**

| Model                     | A1              | A2              | B1              | B2               | B3              | C1                                                                          |
|---------------------------|-----------------|-----------------|-----------------|------------------|-----------------|-----------------------------------------------------------------------------|
| Default Wx, bx from       | Model 1         | Model 1         | Model 1         | Model 1          | Model 10        | Model 1                                                                     |
| Data                      | test 21c        | test 21c        | test 21c        | test Original    | test 21c        | test Original, test 21c, test 14c                                           |
| Predict and Learn         | 0               | 0               | 1               | 1                | 1               | 2                                                                           |
| Supervisor                | Yes             | Yes             | No              | No               | No              | No (One predicted label by three data)                                      |
| batch                     | 10              | 10              | 10              | 10               | 10              | 3                                                                           |
| Alpha                     | 0.1             | 0.0001          | 0.001           | 0.001            | 0.001           | 0.0001                                                                      |
| epoch                     | 60              | 60              | 60              | 60               | 60              | 120                                                                         |
| Accuracy for Original 28c | 93.7%  => 43.8% | 93.7%  => 86.8% | 93.7%  => 66.6% | 93.7%  => 94.1%  | 90.2%  => 88.3% | 93.7%  => 76.7%                                                             |
| Accuracy for 21c          | 52.6% => 95.6%  | 52.6% => 87.3%  | 52.6% => 45.4%  | 52.6% => 53.5%   | 90.9% =>90.7%   | 52.6% => 75.1%                                                              |

<BR>

***a. Supervised Learning Test Results (Model A1 - A2)***

Using Model 1, a new dataset, 21c, is applied to emulate the scenario where "data changes over time"; in this case, the digits become smaller. Since this is supervised learning, the model can access the correct data labels. As shown in the graph below, Model A1 gradually adjusts to the new 21c dataset, achieving 95.6% accuracy by epoch 60. However, the model seems to forget the original dataset, with accuracy dropping to 43.8% at epoch 60. By adjusting the alpha value (learning rate) from 0.1 to 0.0001, the accuracy for both the original dataset and the 21c dataset improves to approximately 86%. This improved performance in Model A2 is superior to Model A1, where the crossover point occurred around epoch 1 or 2, with an accuracy of roughly 75%.

<BR>

<img src="https://github.com/user-attachments/assets/f735e3c6-d098-42ab-955f-99997a95cef0" style="width: 50%; height: 50%;" />

<BR>
<BR>

Accuracy Improvements Over Epoch (Model A1 and A2)

- The model becomes familiar with the new 21c dataset, and with an appropriate alpha value (0.0001), it also performs better on the original dataset.
<p float="left">
  <img src="https://github.com/user-attachments/assets/2eb43c1c-d816-4272-b72f-332c77d4d4da" alt="Image 1" width="500" />
  <img src="https://github.com/user-attachments/assets/b035b010-42ee-4e38-a0f3-443bf50f0c13" alt="Image 2" width="500" />
</p>


<BR>

***b.	Unsupervised Learning (1) Test Results (Model B1 - B3)***

Using Model 1 again, the new 21c dataset is applied to simulate the scenario where "data changes over time." In unsupervised learning, the model does not have access to correct data labels and retrains itself based on self-predicted labels. As shown in the graph below, Model B1 attempts to adjust to the new 21c dataset. However, since the initial accuracy of the original model on the 21c dataset is quite low (52.6%), self-training does not lead to improvements in accuracy for the 21c dataset; in fact, there is a slight degradation. On the other hand, when using Model 10, which was trained on both the original dataset as well as 21c and 14c, self-training with the 21c dataset does not significantly degrade performance on either the original or 21c datasets, though a slight degradation is still observed. This suggests that the model requires correct feedback (i.e., accurate data labels) during training on new datasets in order to improve effectively.

<BR>

<img src="https://github.com/user-attachments/assets/4f9a59d3-02a2-4dc5-ac31-50709441ddce" style="width: 50%; height: 50%;" />

<BR>
<BR>

Accuracy Improvements Over Epoch (Model B1 and B3)
-	Model B1 maintains a similar range of accuracy for the new 21c dataset with no significant improvement. Model B3, which was trained on the original, 21c, and 14c datasets, shows only minimal deterioration.

<p float="left">
  <img src="https://github.com/user-attachments/assets/99dab6b7-1b43-4fb1-afbd-e51f78cc9f5b" alt="Image 1" width="500" />
  <img src="https://github.com/user-attachments/assets/6ad6d5a5-21d4-454a-b840-a32c191f9f05" alt="Image 2" width="500" />
</p>

***c.	Unsupervised Learning (2) Test Results (Model C1)***

Finally, using Model 1, new datasets 14c and 21c, along with the original dataset, are applied to simulate the scenario where "data come closer in sequence." Even though this is unsupervised learning, the model can use the predicted label with the highest accuracy across a sequence of three data predictions and assume all three data points represent the same object. In most cases, the predicted label from the original dataset (with the largest digit size) would have the highest accuracy. As shown in the graph below, Model C1 successfully improved accuracy for both the new 21c dataset (from 52.6% to 75.1%) and the new 14c dataset (from 24.6% to 74.9%) at the cost of accuracy of the original dataset (from 93.7% to 76.7%).

<BR>

<img src="https://github.com/user-attachments/assets/3518d41f-0bc5-4765-a9fa-99d25e53f831" style="width: 50%; height: 50%;" />

<BR>
<BR>

Accuracy Improvements Over Epoch (Model C1)
-	Model C1 successfully improves the accuracies for new datasets (21c and 14c) by its won prediction mainly from the original dataset.

![image](https://github.com/user-attachments/assets/8a8ef34b-8d3c-4ac4-ac75-c5f35e62b504)


Predictions for the first 20 data points of the original, 21c, and 14c datasets at Epoch 0 and 120 (Model C1)
-	There are 18 errors at Epoch 0, and 13 errors (7 original errors + 6 new errors) and 11 corrections at Epoch 120.

![image](https://github.com/user-attachments/assets/094ee1c0-c258-4f3a-8945-778f564c5f0b)

![image](https://github.com/user-attachments/assets/e38eb7b4-bf5f-4cf1-9692-aea13d265be5)


## 5. Conclustion

### Nural Network Model on the Fly Learning Conculustion

#### Self-training without correct labels does not yield satisfactory results. Correct feedback is essential for improving the model on the fly.

As demonstrated in the Model A1 results, a model with correct label feedback can effectively retrain on the fly. In contrast, the Model B1 results show that a model lacking correct label feedback does not retrain well. However, the Model C1 results indicate that the model successfully adapts to new datasets even without external feedback, using a technique that assumes multiple sequential images represent the same object. In conclusion, when designing a system intended for continuous improvement, it is crucial to establish a mechanism for obtaining feedback from either external sources or through self-assumption techniques.
