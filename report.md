# Some adversarial attempts on MNIST

## 1. Abstract 

This project

1. Trained a model (will be refered as orginal model later) to predict the handwritten digits (MNIST) dataset
2. Collect the flip of points on image, to make the model recognize the image to a wrong number
3. Explain that the greedy approach for collecting flips is reasonable
4. Train the adversarial model based on the collected flips
5. Use the adversarial model to create examples to mislead original model and validate the attack success rate
6. Train a distillation model from original model
7. Compare the recognition on adversarial examples generated from adversarial model for original model and distillation model

## 2. Design of experiment

1. Model on MNIST

   The model for MNIST comes directly from [Keras](https://keras.io/examples/vision/mnist_convnet/). The file locates at `model.py`

   ```
      Model: "sequential"
      _________________________________________________________________
       Layer (type)                Output Shape              Param #
      =================================================================
       conv2d (Conv2D)             (None, 26, 26, 32)        320

       max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0
       )

       conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496

       max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0
       2D)

       flatten (Flatten)           (None, 1600)              0

       dropout (Dropout)           (None, 1600)              0

       dense (Dense)               (None, 10)                16010

      =================================================================
      Total params: 34,826
      Trainable params: 34,826
      Non-trainable params: 0
   ```

2. Flip of points

   There is explanation why to use flips (rather than changing greyscale in certain ways) in [progress report 2](progress.md)(I missed the due date and when I finished it I found only 2 people submitted, so I am just attaching the report along with this final report). 
   After around 7 days running at server (Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz, not any fancy performance CPU, but not bad) collected 7000 sets of points to attck the original model.

   The file used in this part is `run.py`

3. Greedy explanation

   Assume if we have the points, we want to prove that the greedy approach (find each flip of points that lower the original confidence most) will yield the least amount of flips to mislead the original model. We design a small experiment to explain that greedy approach will produce __reasonable small amount__ of flips to mislead the original model.

   Assume we have points, we pin 2 random points from the sequential lowest to the collection, then use the same approach to find the flips to lower the original model confidence. Then we compare the random pinned points to the original sequential points, to see the difference.

   The result is that, if we pin 2 random points of sequential points and find the flips, the same points versus the different points is 332-140 (for small amount of tests as finding these points is very time-consuming). The points of same amount of points and less versus larger is 68-7. That is, with our greedy approach, it will produce reasonably least amount of points to lower the original model confidence.

    The source code for testing locates at `greedy_check.py` and `greedy_compare.py`

4. Train adversarial model  
   
   From above we have collected the points to attack the original model, we want to train an adversarial model on the original model. An observation from the points we collected is that on average, we need 6.7 points to make original model recognize the image as wrong number.

   I tried train the model to predict x positions (tried 5 and 7) that will lower the original model confidence, that is the model input shape to output shape `28*28*1 -> x*2*1` However, after many attempts, I can't managed to train the model with a reasonable accuracy, most of the time the models will converge the accuracy at 0.114, I am not sure why.

   Then, I tried to train a model that have the points marked as 1 laying on empty 28*28 zero array, then use similar layers in original model, to creare the adversarial model. 

    ```
      Model output (None, 28, 28, 1)
      Model: "sequential"
      _________________________________________________________________
       Layer (type)                Output Shape              Param #   
      =================================================================
       conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                       
       max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
       )                                                               
                                                                       
       flatten (Flatten)           (None, 5408)              0         
                                                                       
       dense (Dense)               (None, 784)               4240656   
                                                                       
       reshape (Reshape)           (None, 28, 28, 1)         0         
                                                                       
      =================================================================
      Total params: 4,240,976
      Trainable params: 4,240,976
      Non-trainable params: 0
    ```
    
    And it turned out that this approach works, and the accuracy is relatively high than training on the points

    ```
    Epoch 1/5
      44/44 [==============================] - 1s 27ms/step - loss: 0.0299 - accuracy: 0.9793 - val_loss: 0.0073 - val_accuracy: 0.9927
    ```

    I doubt that the convention is (almost) linear and in fact don't need so many layers.

5. To create adversarial examples
    
    The output of such model o a 28*28*1 array is a 28*28*1 array, and the positions range from 0-1, so we need a factor that multiple the position. For example, assume we have 0.13 at position (2,5), we set the multiple to be 20, then it become 2.6 at point (2,5), we limit each point at max 1, so point (2,5) become 1, other points if less than 1 we change that point to 0 (by convert each point to type int). Test run the images and see how big `multiple_factor` will change the recognition from original model, the higher this factor is, our model will need more points to mislead the original model.

    After test on around 2000 images, I found that the average for such multiple factor is 276, and most of the image will share a similar shape for attack. If we feed the output from the adversarial model to heatmap, we can see that at which position attack will more likely happen (at least the model think so)
    For example, if we have a image that looks like this  
    <img src="assets/output1.png"> <img src="assets/output.png"><img src="assets/output2.png">  
    
    And the image shown above is originall recognized as 4, now it is recognized as 9.

6. Train distillation model
