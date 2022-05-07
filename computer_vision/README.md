## Overview

*deep_perception* provides demonstration code that uses deep learning models to precieve the world using 2D, 3D computer vision systen and fusion of 2D and 3D computer vision system 


### Face Estimation Demo

First, try running the face detection demonstration via the following command:

```
roslaunch deep_perception robot_detect_faces.launch 
```

RViz should show you the robot, the point cloud from the camera, and information about detected faces. If it detects a face, it should show a 3D planar model of the face and 3D facial landmarks. Deep Learning models comes from build in models and will use other research models.

You can use the keyboard_teleop commands within the terminal that you ran roslaunch in order to move the robot's head around to see your face.


```
             i (tilt up)
	     
j (pan left)               l (pan right)

             , (tilt down)
```

Pan left and pan right are in terms of the robot's left and the robot's right.

### Object Detection Demo

Second, which uses the tiny YOLO v3 object detection network (https://pjreddie.com/darknet/yolo/). RViz will display planar detection regions. Created models with be use for object detection with 2D and 3D Computer vision.

```
roslaunch deep_perception robot_detect_objects.launch
```

### Body Landmark Detection Demo

Third, try running the body landmark point detection demo. The deep learning model comes from the Open Model Zoo (https://github.com/opencv/open_model_zoo). RViz will display colored 3D points on body landmarks. The network also provides information to connect these landmarks, but this demo code does not currently use it.


```
roslaunch deep_perception robot_detect_body_landmarks.launch 
```

### Nearest Mouth Detection Demo

Finally, try running the nearest mouth detection demo. RViz will display a 3D frame of reference estimated for the nearest mouth detected by the robot. Sometimes the point cloud will make it difficult to see. Disabling the point cloud view in RViz will make it more visible.

We have used this frame of reference to deliver food near a person's mouth. This has the potential to be useful for assistive feeding. However, use of this detector in this way could be risky. Please be very careful and aware that you are using it at your own risk.

A less risky use of this detection is for object delivery. stretch_demos has a demonstration that delivers an object based on this frame of reference by holding out the object some distance from the mouth location and below the mouth location with respect to the world frame. This works well and is inspired by similar methods used with the robot EL-E at Georgia Tech [1]. 


```
roslaunch deep_perception robot_detect_nearest_mouth.launch 
```

## References

[1] Hand It Over or Set It Down: A User Study of Object Delivery with an Assistive Mobile Manipulator, Young Sang Choi, Tiffany L. Chen, Advait Jain, Cressel Anderson, Jonathan D. Glass, and Charles C. Kemp, IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN), 2009. http://pwp.gatech.edu/hrl/wp-content/uploads/sites/231/2016/05/roman2009_delivery.pdf


## License

For license information, please see the LICENSE files. 
