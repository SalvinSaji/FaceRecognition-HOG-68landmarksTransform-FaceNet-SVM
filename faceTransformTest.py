import sys
import dlib
import cv2
import os
import align_dlib

predictor_model = "shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = align_dlib.AlignDlib(predictor_model)

# Take the image file name from the command line
file_name = sys.argv[1]

# Load the image
image = cv2.imread(file_name)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
	pose_landmarks = face_pose_predictor(image, face_rect)

	# Use align_dlib to calculate and perform the face alignment
	alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)

	# Save the aligned image to a file
	if not os.path.exists('outdirtest'):
		os.mkdir('outdirtest')

	if not os.path.exists('outdirtest/test'):
		os.mkdir('outdirtest/test')
		
	cv2.imwrite("outdirtest/test/aligned_face_{}.jpg".format(i), alignedFace)
