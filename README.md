# 16432 Designing Computer Vision Apps - Final Project

# Interactive Augmented Reality of Tumbler

Kung-Hsien (Sam) Yu, YenJu (Jocelyn) Liu


## **Summary**

In this project, we will implement a virtual rendered tumbler which can interact with people in iPad Air 2. The interaction means people can push the toy virtually, and the extent of wobbling depends on the forces. This is a proof-of-concept application, because the tumbler can be easily replaced by other objects, and the interactions can be different based on applications. We choose the tumbler due to its fun and comforting characteristic.


## **Background**

Due to the real-time characteristic, it is important to run it as fast as possible. Therefore, we will leverage OpenGl ES whenever it is possible, and make a good use of computational components (CPU and GPU) at all times. Because the virtual rendering requires an accurate estimate of the device’s location, in addition to the high-speed camera, the internal sensors like Gyro sensors can be used to make the pose estimator more robust.


## **The Challenge**

In AR applications, it poses many difficult problems, and can’t be easily solved by OpenCV library. The challenges include rendering objects on an certain position, tracking objects that moves due to relative motions between the camera and objects, implementing a robust pose estimator, and estimating external forces on the objects. To achieve a robust and high FPS application, we need to modify novel CV algorithms, and combine them in an elegant way.


## **Goals & Deliverables**

**Plan to achieve**

Virtually rendering the tumbler, and creating wobble effects based on estimated external forces

**Hope to achieve**

1. Rendering the same virtual tumbler on multiple devices
2. cooperate with internal sensors to achieve a robust AR applications

**Evaluation**

Since there is no ground truth, it is hard to quantize the performance. Luckily, our performance can be easily judged by human eyes, and we will have a time portfolio on each task like tracking objects, and pose estimator to show how well we can do when optimizing tasks.


## **Schedule**
- 3/27-4/02 Create the 3D model of virtual tumbler and implement basic UI
- 4/03-4/09 Virtual rendering of tumbler
- 4/10-4/16  Identify pose and position of extra forces
- 4/17-4/23  Generate corresponding extent effect on the virtual tumbler
- 4/24-4/30 Testing and Final Tuning
- 5/01-5/04  Demo Preparation
