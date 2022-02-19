import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# CREATING OBJECTS FOR FACE RECOGNATION AND GET INFORMATION ABOUT POINTS IN FACES
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, static_image_mode=True)

# MACHINE LEARNING MODEL WE USE
model = LogisticRegression()

# CREATING DATAFRAME FOR MACHINE LEARNING
# GETTING FOLDER NAME AND RUN OVER ALL THE PICTURES AND COLLECT ALL DATA WE NEED
# WHEN FINISHED FUNCTION RENTURNS READY DATAFRAME
def create_dataframe(dir_name):
    main_path = os.path.join(os.getcwd(), "images", dir_name)
    data_for_df = []

    for root, dirs, files in os.walk(main_path):
        for file_name in files:
            data = []
            # print(os.path.join(root,file_name))
            path = os.path.join(root,file_name)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = faceMesh.process(img_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for i in range(465):
                        x = face_landmarks.landmark[i].x
                        y = face_landmarks.landmark[i].y
                        z = face_landmarks.landmark[i].z
                        data.append(x)
                        data.append(y)
                        data.append(z)
                        # data.append([x, y, z])

                data.append(dir_name.replace("_"," "))
                data_for_df.append(data)

    dataframe = pd.DataFrame(data_for_df)
    dataframe = dataframe.drop_duplicates()

    return dataframe


def predict(file_name):
    path = os.path.join(os.getcwd(), "images", file_name)
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = []  # FOR FACE POINTS
    results = faceMesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:  # GETTINT FACE LANDMARK AND PUTTING IT IN THE LIST
            for i in range(465):
                x = face_landmarks.landmark[i].x
                y = face_landmarks.landmark[i].y
                z = face_landmarks.landmark[i].z
                data.append(x)
                data.append(y)
                data.append(z)
        data = np.array(data)
        data = data.reshape(1, -1)

        predict_result = model.predict(data)[0] #
        img = image = cv2.putText(img, predict_result, (25,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)  # DISPLAING THE NAME ON THE PICTURE
        # DISPLAING FACE LANDMARKS
        mpDraw.draw_landmarks(img,
                              face_landmarks,
                              mpFaceMesh.FACEMESH_CONTOURS,
                              mpDraw.DrawingSpec(color=(100,0,0), circle_radius=3, thickness=1),
                              mpDraw.DrawingSpec(color=(255,150,0), thickness=1))

        cv2.imshow(file_name.split(".")[0], cv2.resize(img, (0,0), fx=0.5, fy=0.5))
    return data


def main():
    messi = create_dataframe("Leo Messi") # CREATING DATAFRAME FOR MESSI
    ronaldo = create_dataframe("Cristiano Ronaldo") # CREATING DATAFRAME FOR RONALDO
    data = pd.concat([messi, ronaldo], ignore_index=True) # MERGE ALL DATAFRAMES TO ONE

    X = data.iloc[:,0:1395]  # COLUMNS OF FACE POINTS
    y = data[1395] # COLUMN OF PRESON NAME
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)  # SPLIT ALL DATA TO 2, ONE FOR TRAINING THE MACHINE LEARNING AND THE OTHER FOR TEST THE MACHINE LEARNING
    model.fit(X_train,y_train) # TRAIN MODEL

    X_train_prediction = model.predict(X_train) # PREDICTION THE TRAINING DATA
    training_data_accuracy = accuracy_score(X_train_prediction, y_train)
    print('Accuracy: ' + str(training_data_accuracy*100) + "%")

    X_test_prediction = model.predict(X_test) # PREDICTION THE TEST DATA
    training_data_accuracy = accuracy_score(X_test_prediction, y_test)
    print('Accuracy: ' + str(training_data_accuracy*100) + "%")

    predict("pic1.jpg")
    predict("pic2.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()