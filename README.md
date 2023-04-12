# FaceRecognition-HOG-68landmarksTransform-FaceNet-SVM
A Face Recognition system to take attendance from the class

## Steps to use 

# Step 1
download the pre-trained model shape_predictor_68_face_landmarks.dat. As it is used to detect the 68 landmarks of the face. Also install all module required as they pop up.

# Step 2
run faceTransformTrain.py after properly placing the images to train (as given) in the train_img directory. It will detect faces and store their transformed versions into a directory(which will be created) named outdirtrain

# Step 3 
run embedtrain.py. It will read the create 128 embeddings of the image and label it which will be stored in the embeddings.npy and labels.npy respectively 

# Step 4 
run faceTransformTest.py followed by the dir to the image to recognize in the terminal. This will detect the images of faces from the image passed and generate the transformed versions of all the faces which will be stored in a directory outdittest.

# Step 5
run embedTest.py. It will create embeddings of the faces to be recognized from the outdittest, and store it in test_embeddings.npy

# Step 6
run svmclassify.py. It will train a SVM classifier based on embedddings.npy and labels.npy and gives the predicted output by taking the embeddings from test_embeddings.npy
