import cv2
import dlib

detector_face = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
video_capture = cv2.VideoCapture(0)

while True:

    ok, frame = video_capture.read()
    deteccoes = detector_face(frame)

    for face in deteccoes:
        cv2.rectangle(frame, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()), (0, 255, 0), 2)
        print(face.confidence)
    cv2.imshow('Video:', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
