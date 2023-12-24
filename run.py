import cv2
from hand_tracker_nms import HandTrackerNMS
import extra
import joblib


WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

connections = extra.connections
int_to_char = extra.classes

detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

gesture_clf = joblib.load(r'models\\gesture_clf.pkl')

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)


word = []
letter = ""
staticGesture = 0

while True:
    hasFrame, frame = capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bboxes, joints = detector(image)
    if points is not None:
        extra.draw_points(points, frame)
        pred_sign = extra.predict_sign(joints, gesture_clf, int_to_char)
        if letter == pred_sign:
            staticGesture += 1
        else:
            letter = pred_sign
            staticGesture = 0
        if staticGesture > 6:
            word.append(letter)
            staticGesture = 0

    if points is None:
        try:
            if word[-1] != " ":
                staticGesture += 1
                if staticGesture > 6:
                    word.append(" ")
                    staticGesture = 0
        except IndexError:
            print("list 'word' is empty")

    extra.draw_sign(word, frame, (50, 460))

    # out.write(frame)
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 8:
        try:
            del word[-1]
        except IndexError as e:
            print(e)


capture.release(1)
cv2.destroyAllWindows()
