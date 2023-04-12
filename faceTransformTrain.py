import sys
import dlib
import cv2
import os
import align_dlib

def find_faces(outdir,indir):
    predictor_model = "shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = align_dlib.AlignDlib(predictor_model)

    for file in os.listdir(indir):
        ind = 0
        for file_name in os.listdir(f'{indir}/{file}'):
            ind = ind + 1
            print(file_name)
            image = cv2.imread(f'{indir}/{file}/{file_name}')

            # Run the HOG face detector on the image data
            detected_faces = face_detector(image, 1)

            print("Found {} faces in the image file {}".format(len(detected_faces), file_name))
            # Loop through each face we found in the image
            for i, face_rect in enumerate(detected_faces):

                # Detected faces are returned as an object with the coordinates 
                # of the top, left, right and bottom edges
                print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

                # Get the the face's pose
                pose_landmarks = face_pose_predictor(image, face_rect)

                # Use openface to calculate and perform the face alignment
                alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)

                # Save the aligned image to a file
                if not os.path.exists(outdir):
                    os.mkdir(outdir)

                if not os.path.exists(f'{outdir}/{file}'):
                    os.mkdir(f'{outdir}/{file}')
                
                cv2.imwrite(f"{outdir}/{file}/aligned_face_{ind}.jpg", alignedFace)
                

find_faces('outdirtrain','train_img')