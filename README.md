"# How-to-build-plant-detector-using-tensorflow-object-detection-model" 
# How-to-build-plant-detector-using-tensorflow-object-detection-model
"# How-to-build-plant-detector-using-tensorflow-object-detection-model" 
This github repository also included these few file as below :

generate_tfrecord.py 
this file is combine two file  xml_to_csv.py and generate_tfrecord.py 
It can straigh away from straight away from xml to tf_record.
Exmaple commands in commands prompts :

python generate_tfrecord.py --image_input=E:/healthy/  --output_path=C:/tensorflow1/models/research/object_detection/train.record --folder_name=train



Object_detection_image.py
change PATH_TO_IMAGE 
change NUM_CLASSES


Object_detection_video.py
change PATH_TO_VIDEO 
change NUM_CLASSES


Object_detection_webcam.py
change NUM_CLASSES
