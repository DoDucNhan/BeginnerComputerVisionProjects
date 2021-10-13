# Beginner Computer Vision Projects

This is a collection of beginner computer vision projects. The idea of all these projects is [here](https://data-flair.training/blogs/computer-vision-project-ideas/).

## List of projects
- Edge detection
- Photo sketching
- Detecting contours
- Mosaic Generator
- Panorama Stitching
- Object Tracking with Camshift Algorithm (In progress)

## Setups and run project
1. Clone the repository
```
git clone https://github.com/DoDucNhan/BeginnerComputerVisionProjects.git
```

2. Install dependencies
```
pip install -r requirements.txt
```

## Edge detection, Photo sketching, Detecting contours, Mosaic Generator
The first 4 projects use [**Streamlit**](https://streamlit.io/) to help users easily perform tasks which are also the names of the projects. All you need to do is run the command.
```
python cv-webapp.py 
```
### Notes
The *Mosaic Generator's* execution time will be longer for large sized images and you will need a wide variety of images to be able to create a good mosaic photo.

## Panorama Stitching
In order to execute the panorama stitching script, you need to run the following command
```
python panorama.py -i stit_folder -o output.jpg -c 1
```

- `-i`: path to input directory of images to stitch.
- `-o`: path to the output image.
- `-c`: whether to crop out largest rectangular region.

## References
- https://towardsdatascience.com/generate-pencil-sketch-from-photo-in-python-7c56802d8acb
- https://learnopencv.com/contour-detection-using-opencv-python-c/
- https://medium.com/m/global-identity?redirectUrl=https%3A%2F%2Fmedium.datadriveninvestor.com%2Fhow-to-build-your-mosaic-image-from-scratch-using-python-123e1934e977
- https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
