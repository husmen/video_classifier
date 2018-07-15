# video_classifier
program to classify videos into security footage or not

# Summary
The aim of this project is to implement an efficient program that could determine whether a video is from a security camera or not. To achieve that we rely on the fact that security cameras are stationary, hence their footage should have more similar frames than normal videos. This means that comparing consecutive frames in a video is enough to determine whether it is a security footage or not. As for the algorithm to quantify the similarity between frames, the Structural Similarity Index has been chosen. This algorithm requires more computational resources but it provides more accurate results than the Mean Squared Error. The latter - while simple to implement - is not highly indicative of perceived similarity. Structural similarity aims to address this shortcoming by taking texture into account [1][2].

[1] Zhou Wang; Bovik, A.C.; ,”Mean squared error: Love it or leave it? A new look at Signal Fidelity Measures,” Signal Processing Magazine, IEEE, vol. 26, no. 1, pp. 98-117, Jan. 2009.

[2] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
