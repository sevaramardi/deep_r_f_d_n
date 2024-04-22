# deep_r_f_d_n
In this paper we introduce the lightweight image super-resolution for IoT devices using new Deep Residual Feature Distillation Network. We are inspired by RFDN model the winner of AIM 2020 efficient super-resolution challenge. In our work we provide modified model RFD block into multi-kernal depthwise-separable convolution blocks(MKDCB). In the Figure 1 is procided MKDC block with list of kernel sizes look at mkdcb.py for more details. This approach helps us to find out best kernel size for capturing the image. Notice you can experiment with several other sizes.

![mkdcb](https://github.com/sevaramardi/deep_r_f_d_n/assets/122605318/8cfe7cd0-878e-4ecd-b8e7-c6206c2500d2)

Our main goal is decrease validation time[ms], number of params[M], flops[G] and memory consumption[M]. To achieve this results we used depthwise-separable convolution instead of standerd convolution layers. As shown in the table we compared the results of our model with the state-of-the-art methods on benchmark datasets. 
![table](https://github.com/sevaramardi/deep_r_f_d_n/assets/122605318/82cf8e47-d46e-40ed-b65d-5919e3eea1f0)

The efficacy of the model shows how the model is suitable for IoT devices. Calculation process is explained in the paper in Section 3.3 Embedded device profiling and formulation.
In the repository deep_r_f_d_n, are 3 major .py files for
running the code: main.py, mkdc_block.py, are for the training the
model and test.py for testing it.
Read the full paper you can in https://www.sciencedirect.com/science/article/pii/S0950705123010912









 
