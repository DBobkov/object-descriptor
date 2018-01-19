---
layout: default
---

# Abstract

Object retrieval and classification in point cloud data is challenged by noise, irregular sampling density and occlusion. To address this issue, we propose a point pair descriptor that is robust to noise and occlusion and achieves high retrieval accuracy. We further show how the proposed descriptor can be used in a 4D convolutional neural network for the task of object classification. We propose a novel 4D convolutional layer that is able to learn class-specific clusters in the descriptor histograms. Finally, we provide experimental validation on 3 benchmark datasets, which confirms the superiority of the proposed approach.


# Main contributions

1. We present a novel 4D convolutional neural network architecture that takes a 4D descriptor as input and out performs existing deep learning approaches on realistic point cloud datasets.

2. We design a handcrafted point pair function-based 4D descriptor that offers high robustness for realistic noisy point cloud data.


 
For full-text of the paper, see <a href="https://doi.org/10.1109/LRA.2018.2792681">IEEE version</a>. 

Video here 
<iframe width="635" height="360" src="res/video_preliminary.mp4" frameborder="0" allowfullscreen></iframe>


# Overview of the pipeline

<img src="res/teaser_figure_revision.png" alt="Overview of the pipeline" width="1250">
Fig. 1. Overview of the proposed object classification pipeline that is a combination of a novel handcrafted descriptor and a 4D convolutional neural network (CNN). For details on the network architecture and layer dimensions, see Fig. 6 and Table I. Here, FC denotes a fully connected layer. 

<img src="res/neural_network_details_very_small.png" alt="Neural network architecture details" width="864">
Architecture of the proposed 4D neural network. See Table I for more details on the dimensions.


# Results

<img src="res/retrieval_results.png" alt="Retrieval results" width="1386">
TABLE II. Retrieval performance of the handcrafted descriptors. The mean value is given in the corresponding column, while
the standard deviation is given in brackets. Best performance is shown in bold.

<img src="res/deep_learning_results.png" alt="Retrieval results" width="901">
Table III. Classification performance of deep learning approaches using 2D, 3D and 4D convolutional layers.

<img src="res/intuition_deep_network.png" alt="Retrieval results" width="665">
Fig.  Descriptor and 4D neural network responses for the object table in the ScanNet dataset. Left: descriptor values. Middle: response of the first filter in the first layer. Right: filter response in the second layer. The rows show slices of the fourth dimension. Transparent bins correspond to constant offset values for the response (or 0 for the descriptor values), colored bins - to varying values. The bins are colored so that low values are shown in blue color, while high in red.


# References

1. C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

2. E. Wahl, U. Hillenbrand, and G. Hirzinger, “Surflet-pair-relation histograms: a statistical 3d-shape representation for rapid classification,” in Proceedings of IEEE International Conference on 3-D Digital Imaging and Modeling (3DIM), 2003, pp. 474–481.



# Contact
For any questions or inquiries, please contact Dmytro Bobkov at dmytro.bobkov@tum.de with a subject "Object Descriptor RAL".
