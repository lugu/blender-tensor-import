## Todo

* tensorflow: create input data
* tensorflow: test hyper parameter with placeholder
* tensorflow: use a cross validation test

* blender: automate the generation of images

* papers: read caffe paper
* papers: print nature article on neural networks
* papers: read tutorial on convolutional networks: http://cs231n.github.io/convolutional-networks/
* papers: print LeNet papper
* papers: read LeNet papper
* papers: read LeNet caffe tutorial:  http://caffe.berkeleyvision.org/gathered/examples/mnist.html

* caffe: convert images into imput format
* caffe: convert z-images into ground-truth format
* caffe: read flownet model file for caffe
* caffe: draw the flownet model

## Done

* caffe: install caffe
* caffe: build caffe with GPU support (cuda)
* caffe: build flownet caffe directory
* caffe: train on GPU with cuda LeNet example)

* tensorflow: install tensorflow
* tensorflow: read tutorials
* tensorflow: watch video presentation from developpers
* tensorflow: run lenet tutorial
* tensorflow: read documentation
* tensorflow: use tensorboard web ui
* tensorflow: use summaries
* tensorflow: build network graph 
* tensorflow: mnist: visualize kernel
* tensorflow: mnist: overlap experiences with tensorboard logdir
* tensorflow: mnist: plot accuracy along CPU time
* tensorflow: mnist: same graph, training accuracy and test accuracy
* tensorflow: mnist: enable/disable dropout

* papers: print dispnet paper
* papers: print flownet paper
* papers: print mc-cnn paper from LeCun
* papers: read dispnet paper
* papers: read flownet paper
* papers: read mc-cnn paper from LeCun

* blender: install blender
* blender: make a script to generate scenes of cubes
* blender: use the composer to extract only the z-image
* blender: save images to png files


## Definitions

Optical flow: the pattern of apparent motion of
objects. One camera, two images, for each pixel find the movement.

Scene flow: the position and direction of each visual point:
z-distance + direction. Two camera, four images.

Stereo vision: estimation of the z-distance from two camera.

Disparity map: difference in image location of objects seen by two
cameras.

Epipolar geometry: geometry of stereo vision: given a disparity map,
compute the z-distance of each point.

Dropout: disable temporary some neurones to avoid overfitting by
limiting the coupling of the neurones.

Softmax: generate a distribution probability give a set of inputs
weigths. It is a normanlized exponential: for each class, compute the
exponent of the value and divide it by of sum of the exponent of the
other classes.

	softmax(x_i) = exp(x_i) / sum_j ( exp(x_j) )


loss function: often use cross enthropy



## Mease of accuracy

http://vision.middlebury.edu/stereo/

## Accuracy measurement implementation

## Best methods

http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

## Convolutional network

https://github.com/jzbontar/mc-cnn

## Tutorial

Tutorial on convulutional networks:
	http://cs231n.github.io
	http://cs231n.github.io/convolutional-networks/

Introduction to convolutional network:
	http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
	http://colah.github.io/posts/2014-07-Understanding-Convolutions/

Cross-enthropy:
	http://colah.github.io/posts/2015-09-Visual-Information/

Weight initialization (xavier):
	http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization

Imagenet: First massive hit with convolutional networks (imagenet.pdf)


Caffe: Implementation for convolutional networks

Tensorflow: NMIST tutorial

## Blender

Tools to use:
	* Randomize transform
	* Node editor composing (compoosite is the end)
	* set a black background


## Caffe commands

	# caffe dir
	export CAFFE_ROOT=/home/ludo/src/caffe-rc3
	cd $CAFFE_ROOT

	# prepare the data
	./data/mnist/get_mnist.sh
	./examples/mnist/create_mnist.sh

	# visualize the model
	python2 python/draw_net.py examples/mnist/lenet_train_test.prototxt lenet-architecture.png
	feh lenet-architecture.png

	# train the model
	./tools/caffe train -solver examples/mnist/lenet_solver.prototxt

## Caffe opinion

Good to:
* to create complex model
* to make repeatable experiences and change params
* to eliminate implementation issues
* to display networks topology
* to exchange models with others people
* to compare optimization solution
* to study and plot accuracy 
* to switch between GPU and CPU
* to separate implementation from network description

Bad for:
* hand craft optmizations
* input / output format not obvious
* not very well documented
* general machine learning solution (just for neural networks)
