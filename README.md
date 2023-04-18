# Fresh-vs-Rotten-Fruits-Classification

Fruits are some of the tastiest and healthy snacks someone could eat, but unfortunately it has a short life span and can easily go bad quickly. It can be extremely harmful to accidently eat a fruit that went bad as it can affect the human body with serious harm. Moreover, stores that sell rotten fruits can get in serious legal trouble and build a bad reputation between customers and competitors. In this model, we are going to utilize Artificial Intelligence and Machine learning to help us recognize if the fruits are fresh or rotten. This will be helpful to manage large number of fruits sold by supermarkets to quickly recognize if the fruits on display or inventory have gone bad.

The dataset consists of 3GB of 8 different types of fruits and 400 pictures for each type, split equally between fresh and rotten. The model is going to be able to recognize the type of fruit in the image and if it is fresh or rotten. To start off, we will train the model to be able to recognize the type of fruit regardless of the size or angle. That will be accomplished by randomly manipulating the original images by rotating, rescaling, shifting, adjusting brightness , and cropping. This process is known as augmentation, as it will increase the size of the dataset and allow our model to train with more diverse images giving a better recognizing ability and better accuracy overall. Moreover, we will split the dataset into training (80%) and testing (20%) segments, which then will run our dataset input into a neural network model. This will allow the model to self-learn and able to achieve the goal of identifying the type of fruit and recognizing if the fruit is rotten or not successfully. 

A convolutional neural network in a sequential model form to classify 8 types of fruits and distinguish between 
fresh and rotten types of fruits efficiently.

After applying fine-tuning techniques, the model achieved an accuracy rate of 74% on the test set.


Dataset: https://data.mendeley.com/datasets/bdd69gyhv8/1

