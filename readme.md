# Pose Annotator Toolkit

This repo originates from the [BOP]((http://bop.felk.cvut.cz)) benchmark for 6D object pose estimation.
I found it hard and unnatural to move the model by keyboard. 
Thus I modified it to support point picking and then the least square optimation can be conducted before the ICP refinement.


## Installation

### Python Dependencies

To install the required python libraries, run:
```
pip install -r requirements.txt
```

In the case of problems, try to first run: ```pip install --upgrade pip setuptools```


## Manual annotation tool

I have moved the parameter config to the GUI panel. So, simply run:

```
python 6D_pose_annotator_v2.py
```
Then set data_root, data_split, split_type, scene_num and image_num in the right panel. **Note that only by setting image_num can the scene be loaded**.

### Interface:

The keyboard interface is kept but with different key-func pair.
- `I` : up, `K` : down, `J` : left, `L` : right, `U` : inner, `M` : outer

Translation/rotation mode:
- Shift not clicked: translation mode
- Shift clicked: rotation mode

Distance/angle big or small:
- Ctrl not clicked: small distance(1cm) / angle(2deg)
- Ctrl clicked: big distance(5cm) / angle(90deg)

`F` or **LSq Refine** button will call least square optimation to compute a inital pose.

`R` or **ICP Refine** button will call ICP algorithm to do local refinement of the annotation.

## Pyinstaller

Sometimes it will be convenient to use [pyinstaller](https://pyinstaller.org/en/stable/) to bundle the toolkit and all its dependencies into a single package. Just run:
```
pyinstaller 6D_pose_annotator_v2.spec
```
Then you can run the .exe file (double click, or drag it into a cmd window) in the generated `dist` directory.

## File Structure
```bash
pose_annotator
├── 6D_pose_annotator_v2.py      # the main script	
├── 6D_pose_annotator_v2.spec	 # used for pyinstaller
├── resources		             # used for pyinstaller
|   └── ...                      # copy from .../site-packages/open3d
├── readme.md
└── requirements.txt
```