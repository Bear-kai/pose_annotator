# Pose Annotator Toolkit

This repo originates from the [BOP](http://bop.felk.cvut.cz) benchmark for 6D object pose estimation.
I found it hard and unnatural to move the model by keyboard. 
Thus I modified it to support point picking and then the least square optimation can be conducted before the ICP refinement.

https://user-images.githubusercontent.com/10415005/177565091-3b04f5f3-01b3-473b-8a5c-f6e16842ad81.mp4

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

### Notice
- It would be better to use meaningful names for the `.ply` model. The provided `YCB_bop_fake` dataset shows a counter-example that you would have to remember/query the correspondence between the model ID and its category name to add a mesh into the scene. Therefore, it is recommanded to use meanfuling model names and then provide a file named `models_names.json` as shown in `YCB_bop_fake/models/`.
- If you want to test on your own data, just organize them similar as the file structure and image name style of the provided `YCB_bop_fake` dataset.

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

About point picking:
- At least three point pairs are selected.
- For example, we select 3 points (#1~#3) on the added model, and 3 points (#4~#6) on the scene.
- Then their order should be the same, meaning that #1 matches to #4, #2 matches to #5, and #3 matches to #6. 

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
├── 6D_pose_annotator_v2.spec    # used for pyinstaller
├── resources                    # used for pyinstaller
|   └── ...                      # copy from .../site-packages/open3d
├── YCB_bop_fake                 # sample data to test on
|   └── ...                      
├── readme.md
├── open3d.ico
└── requirements.txt
```