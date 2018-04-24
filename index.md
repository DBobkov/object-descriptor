---
layout: default
---

# Abstract

Object retrieval and classification in point cloud data is challenged by noise, irregular sampling density and occlusion. To address this issue, we propose a point pair descriptor that is robust to noise and occlusion and achieves high retrieval accuracy. We further show how the proposed descriptor can be used in a 4D convolutional neural network for the task of object classification. We propose a novel 4D convolutional layer that is able to learn class-specific clusters in the descriptor histograms. Finally, we provide experimental validation on 3 benchmark datasets, which confirms the superiority of the proposed approach.


# Paper preview
<img src="res/thumb-0.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-1.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-2.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-3.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-4.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-5.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-6.png" alt="Overview of the pipeline" width="100">
<img src="res/thumb-7.png" alt="Overview of the pipeline" width="100">


# Main contributions

1. We present a novel 4D convolutional neural network architecture that takes a 4D descriptor as input and outperforms existing deep learning approaches on realistic point cloud datasets.

2. We design a handcrafted point pair function-based 4D descriptor that offers high robustness for realistic noisy point cloud data.

For full-text of the paper, see <a href="https://doi.org/10.1109/LRA.2018.2792681">IEEE version</a>.

<iframe width="846" height="480" src="https://www.youtube.com/embed/kQ6w4xA9VeU" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

Code to be uploaded later.

# Overview of the pipeline

<img src="res/teaser_figure_revision.png" alt="Overview of the pipeline" width="1250">
Fig.  Overview of the proposed object classification pipeline that is a combination of a novel handcrafted descriptor and a 4D convolutional neural network (CNN). Here, FC denotes a fully connected layer. 

<img src="res/neural_network_details_very_small.png" alt="Neural network architecture details" width="864">
Fig. Architecture of the proposed 4D neural network. 


# Results

TABLE I. Retrieval performance of the handcrafted descriptors. The mean value is given in the corresponding column, while
the standard deviation is given in brackets. Best performance is shown in bold.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#999;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 6px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#444;background-color:#F7FDFA;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 6px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#fff;background-color:#26ADE4;}
.tg .tg-s6z2{text-align:center}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-j0tj{background-color:#D2E4FC;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-7k3a{background-color:#D2E4FC;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-9bc1{background-color:#D2E4FC;font-weight:bold;font-family:Arial, Helvetica, sans-serif !important;;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-031e" rowspan="2">Dataset</th>
    <th class="tg-s6z2" rowspan="2">Metric</th>
    <th class="tg-baqh" colspan="5">Descriptor</th>
  </tr>
  <tr>
    <td class="tg-j0tj">OUR-CVFH</td>
    <td class="tg-j0tj">ESF</td>
    <td class="tg-j0tj">Wahl</td>
    <td class="tg-j0tj">EPPF Short</td>
    <td class="tg-j0tj">EPPF</td>
  </tr>
  <tr>
    <td class="tg-031e" rowspan="4">Stanford</td>
    <td class="tg-baqh">Total accuracy (%)</td>
    <td class="tg-baqh">62.79</td>
    <td class="tg-baqh">71.34</td>
    <td class="tg-baqh">75.13</td>
    <td class="tg-baqh">77.26</td>
    <td class="tg-amwm">80.18</td>
  </tr>
  <tr>
    <td class="tg-j0tj">Mean accuracy (%)</td>
    <td class="tg-j0tj">42.91</td>
    <td class="tg-j0tj">54.54</td>
    <td class="tg-j0tj">57.00</td>
    <td class="tg-j0tj">60.53</td>
    <td class="tg-7k3a">64.01</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mean recall (%)</td>
    <td class="tg-baqh">49.90</td>
    <td class="tg-baqh">52.28</td>
    <td class="tg-baqh">57.45</td>
    <td class="tg-baqh">60.16</td>
    <td class="tg-amwm">64.58</td>
  </tr>
  <tr>
    <td class="tg-j0tj">F1-score</td>
    <td class="tg-j0tj">0.437</td>
    <td class="tg-j0tj">0.530</td>
    <td class="tg-j0tj">0.567</td>
    <td class="tg-j0tj">0.601</td>
    <td class="tg-7k3a">0.640</td>
  </tr>
  <tr>
    <td class="tg-031e" rowspan="4">ScanNet</td>
    <td class="tg-baqh">Total accuracy (%)</td>
    <td class="tg-baqh">56.23</td>
    <td class="tg-baqh">53.41</td>
    <td class="tg-baqh">63.72</td>
    <td class="tg-baqh">63.49</td>
    <td class="tg-amwm">65.29</td>
  </tr>
  <tr>
    <td class="tg-j0tj">Mean accuracy (%)</td>
    <td class="tg-j0tj">39.83</td>
    <td class="tg-j0tj">33.69</td>
    <td class="tg-9bc1">45.40</td>
    <td class="tg-j0tj">42.02</td>
    <td class="tg-j0tj">44.95</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mean recall (%)</td>
    <td class="tg-baqh">38.21</td>
    <td class="tg-baqh">32.72</td>
    <td class="tg-baqh">45.94</td>
    <td class="tg-baqh">45.17</td>
    <td class="tg-amwm">47.54</td>
  </tr>
  <tr>
    <td class="tg-j0tj">F1-score</td>
    <td class="tg-j0tj">0.382</td>
    <td class="tg-j0tj">0.327</td>
    <td class="tg-j0tj">0.444</td>
    <td class="tg-j0tj">0.430</td>
    <td class="tg-7k3a">0.457</td>
  </tr>
  <tr>
    <td class="tg-s6z2" rowspan="4">M40</td>
    <td class="tg-baqh">Total accuracy (%)</td>
    <td class="tg-baqh">53.22</td>
    <td class="tg-baqh">65.87</td>
    <td class="tg-amwm">74.41</td>
    <td class="tg-baqh">73.00</td>
    <td class="tg-baqh">73.68</td>
  </tr>
  <tr>
    <td class="tg-j0tj">Mean accuracy (%)</td>
    <td class="tg-j0tj">46.43</td>
    <td class="tg-j0tj">58.91</td>
    <td class="tg-7k3a">67.50</td>
    <td class="tg-j0tj">65.79</td>
    <td class="tg-j0tj">66.43</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mean recall (%)</td>
    <td class="tg-baqh">49.26</td>
    <td class="tg-baqh">59.96</td>
    <td class="tg-amwm">70.33</td>
    <td class="tg-baqh">69.12</td>
    <td class="tg-baqh">69.79</td>
  </tr>
  <tr>
    <td class="tg-j0tj">F1-score</td>
    <td class="tg-j0tj">0.465</td>
    <td class="tg-j0tj">0.588</td>
    <td class="tg-7k3a">0.680</td>
    <td class="tg-j0tj">0.666</td>
    <td class="tg-j0tj">0.671</td>
  </tr>
</table>


Table II. Classification performance of deep learning approaches using 2D, 3D and 4D convolutional layers.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;border-color:#999;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#444;background-color:#F7FDFA;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#999;color:#fff;background-color:#26ADE4;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-vn4c{background-color:#D2E4FC}
.tg .tg-804w{font-family:Arial, Helvetica, sans-serif !important;;text-align:center;vertical-align:top}
.tg .tg-yw4l{vertical-align:top}
.tg .tg-j0tj{background-color:#D2E4FC;text-align:center;vertical-align:top}
.tg .tg-7k3a{background-color:#D2E4FC;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-5hgy{background-color:#D2E4FC;text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l">Dataset</th>
    <th class="tg-baqh">Metric</th>
    <th class="tg-baqh">PointNet</th>
    <th class="tg-baqh">EPPF 2D</th>
    <th class="tg-baqh">EPPF 3D</th>
    <th class="tg-baqh">EPPF 4D</th>
  </tr>
  <tr>
    <td class="tg-vn4c" rowspan="4">Stanford</td>
    <td class="tg-j0tj">Total accuracy (%)</td>
    <td class="tg-j0tj">64.30</td>
    <td class="tg-j0tj">82.01</td>
    <td class="tg-j0tj">81.94</td>
    <td class="tg-7k3a">83.22</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mean accuracy (%)</td>
    <td class="tg-baqh">42.48</td>
    <td class="tg-baqh">64.26</td>
    <td class="tg-amwm">66.37</td>
    <td class="tg-baqh">65.11</td>
  </tr>
  <tr>
    <td class="tg-j0tj">Mean recall (%)</td>
    <td class="tg-j0tj">40.47</td>
    <td class="tg-j0tj">70.88</td>
    <td class="tg-j0tj">60.94</td>
    <td class="tg-7k3a">72.13</td>
  </tr>
  <tr>
    <td class="tg-baqh">F1-score</td>
    <td class="tg-baqh">0.395</td>
    <td class="tg-baqh">0.652</td>
    <td class="tg-baqh">0.665</td>
    <td class="tg-amwm">0.672</td>
  </tr>
  <tr>
    <td class="tg-vn4c" rowspan="4">ScanNet</td>
    <td class="tg-j0tj">Total accuracy (%)</td>
    <td class="tg-j0tj">63.04</td>
    <td class="tg-j0tj">70.39</td>
    <td class="tg-j0tj">70.57</td>
    <td class="tg-7k3a">72.10</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mean accuracy (%)</td>
    <td class="tg-baqh">37.50</td>
    <td class="tg-baqh">38.98</td>
    <td class="tg-804w">44.35</td>
    <td class="tg-amwm">45.70</td>
  </tr>
  <tr>
    <td class="tg-j0tj">Mean recall (%)</td>
    <td class="tg-j0tj">19.53</td>
    <td class="tg-7k3a">63.52</td>
    <td class="tg-j0tj">54.53</td>
    <td class="tg-j0tj">56.58</td>
  </tr>
  <tr>
    <td class="tg-baqh">F1-score</td>
    <td class="tg-baqh">0.209</td>
    <td class="tg-baqh">0.433</td>
    <td class="tg-baqh">0.472</td>
    <td class="tg-amwm">0.488</td>
  </tr>
  <tr>
    <td class="tg-5hgy" rowspan="4">M40</td>
    <td class="tg-j0tj">Total accuracy (%)</td>
    <td class="tg-7k3a">87.01</td>
    <td class="tg-j0tj">81.64</td>
    <td class="tg-j0tj">81.15</td>
    <td class="tg-j0tj">82.13</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mean accuracy (%)</td>
    <td class="tg-amwm">82.08</td>
    <td class="tg-baqh">76.37</td>
    <td class="tg-baqh">75.87</td>
    <td class="tg-baqh">77.05</td>
  </tr>
  <tr>
    <td class="tg-j0tj">Mean recall (%)</td>
    <td class="tg-7k3a">83.48</td>
    <td class="tg-j0tj">77.30</td>
    <td class="tg-j0tj">77.51</td>
    <td class="tg-j0tj">76.99</td>
  </tr>
  <tr>
    <td class="tg-baqh">F1-score</td>
    <td class="tg-amwm">0.824</td>
    <td class="tg-baqh">0.765</td>
    <td class="tg-baqh">0.762</td>
    <td class="tg-baqh">0.769</td>
  </tr>
</table>





<img src="res/intuition_deep_network.png" alt="Retrieval results" width="665">

Fig. Descriptor and 4D neural network responses for the object table in the ScanNet dataset. Left: descriptor values. Middle: response of the first filter in the first layer. Right: filter response in the second layer. The rows show slices of the fourth dimension. Transparent bins correspond to constant offset values for the response (or 0 for the descriptor values), colored bins - to varying values. The bins are colored so that low values are shown in blue color, while high in red.


# References
1. C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

2. E. Wahl, U. Hillenbrand, and G. Hirzinger, “Surflet-pair-relation histograms: a statistical 3d-shape representation for rapid classification,” in Proceedings of IEEE International Conference on 3-D Digital Imaging and Modeling (3DIM), 2003, pp. 474–481.



# Contact
For any questions or inquiries, please contact Dmytro Bobkov at <img src="res/email.png" alt="Email" width="150"> with a subject "Object Descriptor RAL".

Last updated 24.04.2018