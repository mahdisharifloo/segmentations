# yolo_obj_localization

## How to run

### installig Darknet
Darknet is an open source neural network framework written in C and CUDA.
you can find it on there: https://pjreddie.com/darknet/  

**NOTE** : you have to change some code on darkflow so don't follow general installation and look at this path to installing darkflow.   
1. 
```bash 
git clone https://github.com/pjreddie/darknet.git
cd darknet
```
2. go to *darknet/src* and open *image.c* file.
3. find *draw_detections* void function and get into it.
4. find blow codes.
```c 
if(bot > im.h-1) bot = im.h-1;
```
5. add this code under code that in parts 4 
```c 
printf("Bounding Box: Left=%d, Top=%d, Right=%d, Bottom=%d\n", left, top, right, bot);
```
you most see just like this codes : 
```c 
...
if(top < 0) top = 0;
if(bot > im.h-1) bot = im.h-1;
printf("Bounding Box: Left=%d, Top=%d, Right=%d, Bottom=%d\n", left, top, right, bot);
draw_box_width(im, left, top, right, bot, width, red, green, blue);
...
```
6. compile c codes with make.
```bash 
make
```
7. you have to dowload the yolov3-tiny.weights with this link : https://pjreddie.com/darknet/yolo/  
8. after run python codes you can input your darknet installation root path to run the project.

good luck