
The main programming language or tool used in this project is Matlab. Because the purpose of this
project is to test the feasibility and evaluate the accuracy performances of different methods, whether
the ideas and methods can be implemented in a clear and straight-forward manner is more important
than the execution speed and efficiency of the code. And there are some computer vision and deep
learning toolbox specically made for Matlab. VLFeat, an open source computer vision library and
MatConvNet, a MATLAB toolbox implementing Convolutional Neural Networks (CNNs) are used
as two primary toolboxes throughout the project.



# Detailed Process of the Regional Max-Pooling Activations (RMAC) from CNN for CBIR: #

**Weichuan Yin**



##### Important element used:

AlexNet pretrained on ImageNet, conv_5 layer activation

Training Set: Oxford 5K

Testing Set: Paris 6K

MAC

RMAC

AML

L2-Norm

PCA-Whitening

Integral Image: just for speed ?

## Step 1 : Pre-Processing Data

### 1.1. Use Training Set ('Oxford 5K') to learn PCA

Firstly, feed each oxford 5k image in to pretrained AlexNet, and extract the activations from layer conv_5 for each image ( W x H x 256 ). Then from each image's activation, extract ( around 20 ) regional-mac features (each of dimension 256 x 1) and apply L2-Norm to all of them. (There are over 100k regional-mac features at this stage).

### 1.2. Calculate the PCA element from the training set

From the above extracted & computed features from training set. Compute the covariance matrix for all 100k regional-mac feautres, and calculate the all the 256 means from each feature dimension, as well as 256 eigenvalues and eigenvectors. This is the preparation of the pre-processing stage done.

### 1.3. Feed Testing Set ('Paris 6K') to CNN

Feed each paris 6k image in pretrained AlexNet, and extract the activations from layer conv_5 for each image ( W x H x 256 ). Then from each test image's activation: 

1. extract ( around 20 ) regional-mac features (each of dimension 256 x 1) :
2. keep all the raw ( W x H x 256 ) conv_5 layer activations for all test images ( to prepare for localization and reranking after the inital filtering stage).

### 1.4. Pre-processing the extracted test regional-mac feautres

For each regional-mac features:

1. L2 norm it;
2. PCA-withening it (based on the mean, eigenvectors calculated from the training set);
3. L2 norm it again;
4. For each image, sum all regional mac to form a single RMAC;
5. L2 norm it once again.



## Step 2: Prepare the Query Image 

Crop the query image based on the bounding box given in the ground truth file.

Then for each cropped query object image, extrac their regional-mac features, L2-norm them, apply PCA-withening, L2 norm again, and sum them up and L2 norm once again to get a final RMAC feature.

Additionally, keep the simple MAC feature for each cropped query object's activation (and also L2 norm it) to prepare for the further localization stage.



## Step 3: Initial Retrieval (Filtering Stage)

Since the RMAC vector for each image is L2-normed, simply use cosine distance ( dot product ) to calculate the distance between each cropped query object and the database images. 

For each query, rank its distance w.r.t. all the databased images to get the initial retrival results and calucate the mAP from it.



## Step 4: Localization 

**Localization is done by AML on Query's L2-normed MAC**

For each query object, firstly calculate the object aspect ratio. Also use the top 1000 retrieval results to keep for the localization and reranking.

For these top 1000 retrieval images, get their raw ( W x H x 256 ) conv_5 layer activations and quantize these raw activations in the same manner as file compression.

Use AML ( approximate max-pooling localization) algorithm in the paper to locate the most-matched regions in the whole retrieval image w.r.t. the query object's simple MAC (L2-normed) feature. And this return a bounding box in the conv_5 layer activation domain.



## Step 5: Reranking to top 1000s

**Reranking is done based on Query's withened RMAC & Database's withened RMAC **

Extract the bounding box specified conv_5 layer activations for each top 1000 retrival images, and compute the RMAC vector for these AML-specified activation regions (L2, PAC Withe, L2, Sum & L2 need to all applied once again).

Therefore for each top 1000 retrieval result, we get another RMAC vector specifying on the region of interest determined by AML w.r.t. the query object. Cosine distance is calculated bewtween them again to rerank in the top 1000 retrival results



## Step 6: Query Expansion

 Re-ranking brings positive images at the very top ranked positions. Then,we collect the 5 top-ranked images, merge them with the query vector, and compute their mean. These are all done in the (L2 normed, PCA withened) RMAC domain. So, we get a new vector for the query, which is the mean from  query object and another 5 top ranked retrieved objects. The RMAC feature of this is dot product with the top 1000 AML determined regions.(Finally, the similarity to this mean vector is adopted to re-rank once more the top 1000  images).
