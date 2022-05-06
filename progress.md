# Term project progress report 2

### Continued from last report

I tested 3 methods out on how to modify the images  
The tests are conducted on MNIST dataset comes along with Keras, each image size is 28*28*1 
On one point,

*   Change the greyscale within 0-1, step by 0.1
*   Change the greyscale to a random number within 0-1, repeat 10 times
*   Flip the greyscale x -> 1-x

Among these 3 approaches on small amount of images (each around 100), it turns out that the third method has the best efficiency and speed, and it leads to a intuition that on each point
__the largest impact to prediction happen when greyscale changes to 0 or 1__

### Current progress

Program is running on computer, runs slower than expected with CPU.
Used the third method on 28*28 points, collected data on around 2k images.  
The observed average flips needed to change prediction is around 6


### Possible directions

There are some work need to complete the project

1.  Demonstrate that the greedy approach will yield the optimal result (will probably be hard for a proof, but can be demonstrated with statistics by running small amount of images)

The exlanation is shown as:
Assume given a image with number 7, with x steps (points) the image is recognize as number 1. 
When we have x points, we pin P1 to be arbitrary one of these x points, and iterate until the image 7 is recognize as different number. The ideal situation should be:

*   The steps required for changing the number is around x
*   The newly recognized number is still 1
*   The points P1', P2' ... should be close to the original P1, P2 ...

2.  There should be some relationship between such security on machine learning and the privacy, but I am not going to dig deeper into this
3.  The vulnerable positions for each number  

May need a simple model to predict the position, or even just matrix  

1.  Inspect vulnerable position in each layer output  
Such as  
[[...][...][...]...]  
-> [[...][...]]  
-> [[0.12, 0.25, 0.34, 0.87, ...]]

For a given number, in each layers, we can check original image and modified image to see whether there are some position in certain layers will have a big impact on the prefiction.