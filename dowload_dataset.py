from roboflow import Roboflow
rf = Roboflow(api_key="NlO0Js6OIvklw9RXfvRN")
project = rf.workspace("senior-design-1a3ye").project("crack-finder-bbzjj")
version = project.version(1)
dataset = version.download("yolov11")
                