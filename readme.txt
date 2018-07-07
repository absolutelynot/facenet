1. Make sure that tensorflow is installed correctly.


2. Download pretrained model from 
https://github.com/davidsandberg/facenet/wiki#pre-trained-models

3. Set environment variable PYTHONPATH to point to the src directory of FaceNet repo. 

export PYTHONPATH=<download_root_for_your_facenet>/facenet/src

4. Use the following command for verification. Some sample images are put in the same directory. 


python recognition_images.py --model <dir_contains_model> --i <path_for_image> --ds <path_for_data_set>



note that <dir_contains_model> is the directory that contains the model rather than the model path.