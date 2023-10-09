This directory contains code that generates partial point clouds from ShapeNet models. To use it:
1. Install [Blender](https://blender.org/download/).
2. Run `blender -b -P render_depth.py  to render the depth images. The images will be stored in OpenEXR format.
3. Run `python3 process_exr.py to convert the `.exr` files into 16 bit PNG depth images and point clouds in the model's coordinate frame.