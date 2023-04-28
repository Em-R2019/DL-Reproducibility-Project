# Reproduction Study
This is a reproduction study of paper in Computer Vision And Machine Learning:
_**It’s Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation**_




## Introduction



## Approach



## Methodology



## Results
Results of our different training sessions are shown in the plots below, associated with a more concise table:

                                                               For batch size 170
<br /> <br />
<img src='pics/scaled_average_170.jpeg' width='400'>----------------------------<img src='pics/notscaled_average_170.jpeg' width='400'>
<br /> <br />
<img src='pics/scaled_best_average_170.jpeg' width='400'>----------------------------<img src='pics/notscaled_best_average_170.jpeg' width='400'>

From the pictures above we can see that most of the later folds have quite an increase in degree of error. When these samples were left out during the leave-one-out cross-validation sequence, the models had a harder time correctly estimating gaze on, something that was also observed by the other group working on the same dataset.

                                                                For batch size 110
<br /> <br />
<img src='pics/scaled_average_110.jpeg' width='400'>----------------------------<img src='pics/notscaled_average_110.jpeg' width='400'>
<br /> <br />
<img src='pics/scaled_best_average_110.jpeg' width='400'>----------------------------<img src='pics/notscaled_best_average_110.jpeg' width='400'>

Similarly to batch size 170 plots, the same samples seem to pose a problem for the trained models.

Table I

|  Batch Size  |  Scaling   | Best Average | Overall average |  SD for best average  | SD for overall average |
|:------------:|:----------:|:------------:|:---------------:|:---------------------:|:----------------------:|
|     170      |    Yes     |   4.15012    |     4.61098     |       0.839635        |        1.05107         | 
|     170      |     No     |   4.78910    |     5.31077     |       0.893591        |        1.12554         | 
|     110      |    Yes     |   4.28300    |     4.70473     |       0.853840        |        1.16285         | 
|     110      |     No     |   5.06958    |     5.60263     |       0.789168        |        1.11690         | 

Table I shows that overall the larger batch size of 170 performed better than the batch size of 110, although not by a significant amount.
However, when tested with a larger batch size of 210, the results were worse than both shown batch sizes. 

These results also show that even with our best scores throughout all epochs averaged, we are still off from the original score of 3.73 by a margin of 0.42, and compared to our overall average over the last epochs, we get a margin of 0.88.

## Analysis



## Discussion

As mentioned in our results section we can see that we are off the original papers degree of error by 0.88 degrees on average and 0.42 at best. We have a couple of suspicions on what could've caused these margins.

### Different AFF net implementation

The original paper did not make available the implementation of AFF-Net that they used. We used an implementation found online from a Github repository linked here: https://github.com/kirito12138/AFF-Net.

There could be a difference between the implementation we have used and what the original paper used, however this is not something that we can know at this time. We suspect that this could be the main reason for our result difference.

### Batch size

There is a batch size difference between the original batch size used by the authors and our own. The authors of the paper did not give any information on the batch size they used, while we used 110, 170 and tested a little on 210. We know that the batch size matters for accuracy as it can determine how much the gradient changes through each step. We took a look at an online discussion on this which goes in more detail:

https://www.kaggle.com/questions-and-answers/185920.

From this discussion we could gather that there could be a significant difference in our results compared to the original authors based on the batch size that they picked.

### Bounding box and seed randomness

The bounding boxes defined in the original implementation are moved randomly between a -30 to 30 pixel range in all directions. We don't do this in our code as it can add some randomness to the results, which may create larger variations in results.

On top of that, the original paper doesn't include whether they set a specific seed for their torch operations or not. We also do not set this seed as we don't think it would introduce a very large difference, however it is worth mentioning that it still is something that can explain part of the margin.


### Changed training procedure

The original paper used multi-gpu parallelization during training. We changed the code to be run on a single-gpu and increased the worker threads. This method could cause some calculating errors to creep up due to the parameter usage from the threads.
