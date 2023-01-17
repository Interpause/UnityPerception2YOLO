# Archived

Obsolete. Use https://github.com/Unity-Technologies/com.unity.perception/blob/main/com.unity.perception/Documentation~/Tutorial/convert_to_coco.md followed by https://github.com/ultralytics/JSON2YOLO instead.

# UnityPerception2YOLO

Converts Unity Perception dataset format to YOLO format.

## What?

Unity's Synthetic Data Package, [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) outputs datasets in their own Perception dataset format, which follows this [schema](https://datasetinsights.readthedocs.io/en/latest/Synthetic_Dataset_Schema.html#synthetic-dataset-schema). They also provide a Python package, [Dataset Insights](https://github.com/Unity-Technologies/datasetinsights) to more conveniently parse their datasets.

In terms of tools available for converting Unity Perception to YOLO, I could only find one: [Roboflow](https://roboflow.com/convert/unity-perception-json-to-yolo-darknet-txt). However, there are no **offline** tools to do so; It is missing from both [CVAT's Datumaro](https://github.com/openvinotoolkit/datumaro) and [PyLabel](https://github.com/pylabel-project/pylabel) as of writing.

Given I currently don't have the time to make a Pull Request, and how simple it is to make a conversion script, I decided to just do it. I also put [PyInstaller](https://pyinstaller.org/en/stable/) in this to make it more convenient for non-coders to convert the dataset :p.
