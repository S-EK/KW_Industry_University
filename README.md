# 표적 식별 딥러닝 모델 성능 비교 시스템 개발
# Deep Learning Model Performance Evaluation System for Target Detection

### Fine애플 팀 : 김민서, 김아영, 김현규, 석은규, 이예진 


##### Abstract: Starting with LeNet, deep learning technology has been continuously developed over time. However, during the process of increasing the deep neural network model’s accuracy and performance, the number of layers built into the deep learning model increased, and this led to the increasing number of parameters. Particularly, hyper-parameters, parameters that cannot be trained by a neural network, require a lot of experience and time to optimize. In this work, we introduce a new system that aids users to easily choose a deep learning model and hyper-parameters optimized and performs best to the user’s input dataset. The system auto-matically compares different convolutional network models that had been optimized with suitable hyper-parameters. Finally, the system shows the user best-performed model with optimized hyper-parameters.
---

### 1. Introduction

  In 1989, Lecun, Yann presented LeNet, the first neural network model. LeNet had several deficiencies including the gradient vanishing problem and XOR problem, however, by solving those drawbacks, deep neural network models had been improved persistently over the last decades. Of its improved performance and accuracy, deep neural network mod-els had been used in various ways including image processing and image classification (CNN). Despite the continuous development of deep learning network, the process of optimizing hyper-parameters for input data is a tricky task that should be kept studied. Unlike other parameters that can be earned by training neural network model, hyper-parameters should be decided by humans and cannot be carried out automatically. Moreover, the final effect of hyper-parameters on the final performance of neural network models is considerable. Thus, the necessity of a system that automatically finds the best-performing neural network model with optimized hyper-parameters for specific imported data is emerging.




### 2. Methodology
  Hyper-parameters are parameters that cannot be earned by the model’s training; therefore, humans must intervene in order to find the right optimized value for input data. Thus, our system allows this complicated task to be progressed completely automatically through several experiments from candidate values given by the user. First, the user inputs image data to the system with classes that the dataset has and some can-didate values that human wants to test out. Then, the sys-tem splits the data into train data, test data, and validation data in the proportion of 4:4:2. Train data gets imported into each CNN model. In this system, we test VGG16, ResNet50, ResNext, and DenseNet. The system imports the pre-trained CNN models for both better accuracy and time performance. Next, each model tests various hyperparameters with can-didate values that the user gave and gets evaluated respectively by three perspectives; accuracy, f1-score, and inference time. Lastly, the system shows the user the result in a separate evaluation index. The user can see the result of the best-performed models. 
  
  
  The candidate CNN models that our system experiments with are VGG16, DenseNet, ResNet50, and ResNext. Our team had considered several various CNN models since the final results can be changed by which model our system compares. Lastly, we decided on those four models from a research paper, “A Transfer Learning Evaluation of Deep Neural Networks for Image Classification”. The paper explains different CNN models’ structures and characteristics, and it also compares their performance in several ways. Those four models we selected were models that were highly performed in the paper’s evaluation. 
  
  
  
  
### 3. Tool Support
  The system consists of four different classes. First ResizeImage class is a module that allows the user to input im-age data and several parameters like image classification classes or candidate hyperparameters. Next, the Comparison System class is for loading different CNN models and evaluating them respectively. Also, there are four different classes for each CNN model (myDenseNet Class, myResNet Class, myResNext Class, myVGG Class) that process fine-tuning task and optimize their hyper-parameters through experimenting every candidate hyper-parameter value that the user gave. Lastly, show_graph class visualizes the result of the evaluation to a user. 


### 3.1 ResizeImage Class
  init() 


•	parameters: input dataset path, size of the image 


•	loads input data from the path and does some preprocessing such as flattening image data


  resizing()


•	resize images and change data into the format of CNN model input


### 3.2 Comparison System Class
  init()


•	set the default value of hyper-parameters


  datasplit()


•	splits input data into train, test, and valid data with a proportion of 4:4:2


  setSystem()


•	loads each pre-trained CNN model


  runSystem()


•	preprocess data and train models with train data 


  getResult()


•	evaluates the model with three evaluation index (accuracy, f1-score, inference time), and deduce the rank-ing result


### 3.3 myVGG19 Class, myDenseNet Class, myResNet Class, myResNext Class
  preprocessing module


•	preprocess image data for each type of CNN model’s input type


  finetuning module (model_vgg16(), model_densenet(), modelResnet50(), model_resnext())


•	optimize hyperparameters through several experiments with given candidate values


  predictResult()


•	predicts test data and return evaluation score for each model and each hyper-parameter value set respec-tively


### 3.4 show_graph Class
  show_accuracy()


•	show the best-performing CNN model’s flow of accuracy in the training phase 


  show_loss()


•	show the best-performing CNN model’s flow of loss in the training phase 


  show_result()


•	show the best-performing CNN model’s f1-score and inference time




### 4. Results
  With the methodology outlined in Sec. 2, Sec.3, we applied this system to CIFAR-10 dataset. Since our candidate CNN models were pre-trained with ImageNet dataset, we judged that CIFAR-10 dataset is suitable for a case study. Be-cause of the limit on our study environment, we used batch size 64, and 10 epochs. The best performed model in accuracy and f1-score was ResNet50 (hyper-parameter: drop-out rate=0.5, learning rate=0.005, optimizer=SGD). From the perspective of inference time, the best-performed model was VGG16 (hyper-parameter: drop-out rate=0.2, learning rate=0.0005, optimizer= ADAM). 




### 5. Conclusions
  This paper introduced a new system that compares different CNN models that have been optimized by specific input data without any human intervention in between. As a result, the system offers the fastest, perfectly target detect-ing model with optimized hyper-parameters. Thus, this system can be used in any way that involves target detecting tasks including the car license plate identification systems or in the national security center. Moreover, our system com-pares different CNN models from three different perspectives; accuracy, f1-score, and inference time. However, addi-tional performance measures and evaluation indexes can be implemented and tested to enhance the performance of this system. 
