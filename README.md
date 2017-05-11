# 16432 Designing Computer Vision Apps - Final Project

# Interactive Augmented Reality of Tumbler

Kung-Hsien (Sam) Yu, YenJu (Jocelyn) Liu


## **Summary**

In this project, we would like to implement a virtual rendered tumbler which can interact with people in iPad Air 2. In our first edition, we have built a prototype of Virtual Tumbler which can detect card corners, track card corners, and project virtual objects onto the cards in real time on iPad Air 2, which can serve as fundamental components for rendering augmented reality virtual tumblers and interacting with them in the future.

## **Background**

OpenCV and Armadillo libraries are used for finding card corners, performing edge detections, obtaining homography matrices, resizing images, matrix operations, and so on.

## **The Challenge**

Nowdays, implementing a virtual augmentation in the mobile devices is still a challenging task, since it needs to consider the power issue, hardware limitations, and real-time requirement.
During implementing the first prototype of our project, we also encountered challenges such as how to effectivly detect cards in a frame, and tracking those card corners efficiently.

## **Goals & Deliverables**

**Plan to achieve**

Projecting virtual objects on cards, and the user can move the camera or
cards, but the objects are still on the correct positions.

**Hope to achieve**
1. Virtually rendering the tumbler, and creating wobble effects based on
   estimated external forces
2. Rendering the same virtual tumbler on multiple devices
3. Cooperating with internal sensors to achieve a robust AR application

**Evaluation**

Since there is no ground truth, it is hard to quantize the performance. Luckily, our performance can be easily judged by human eyes, and we will have a time portfolio on each task like tracking objects, and card detection to show how well we can do when optimizing tasks.

## **Schedule**
- 3/27-4/02 Implement basic UI and learn to use different libraries like OpenCV, Armadillo, and GPUImage
- 4/03-4/09 Card detection and virtual rendering of tumbler
- 4/10-4/16 Card detection and tracking
- 4/17-4/23 Card identification and virtual augmentation
- 4/24-4/30 Testing and Final Tuning
- 5/01-5/04 Demo Preparation
