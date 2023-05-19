## Visual-SLAM

This project was a part of the RBE 549 Course Computer Vision at Worcester Polytechnic Institute under Prof Nitin Sanket. 

The details of the project can be looked at here - https://rbe549.github.io/fall2022/proj/p3/

### Need for normalization

Reference:https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/agartia3/index.html#:~:text=Normalized%20images%20are%20mean%20centred,as%20a%20measure%20of%20performance

Normalizing keypoints centers all points to mean 0 and unit variance. This means that the Eucilidian distance between all points after normalization is constraint to maximum value 1.

Any processing to the points after this affects the point equally. 

While it may sound as if this process creates loss of information, it makes any algorithm robust to variances or noise. This is demonstrated in the article in the link above.


### Demo

<p align="center"> Input samples </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/deveshdatwani/Visual-SLAM/main/P3Data/Imgs.png" width="800">
</p>

<p align="center"> Output example </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/deveshdatwani/Visual-SLAM/main/P3Data/VSfM.png" width="800">
</p>
