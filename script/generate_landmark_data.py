from utils import *


def is_handsign_character(char: str):
    return ord('a') <= ord(char) < ord("q") or char == " "


def label_dict_from_config_file(relative_path):
    with open(relative_path, "r") as f:
        label_tag = yaml.full_load(f)["gestures"]
    return label_tag


class HandDatasetWriter():
    def __init__(self, filepath) -> None:
        self.csv_file = open(filepath, "a")
        self.file_writer = csv.writer(self.csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    def add(self, hand, label):
        self.file_writer.writerow([label, *np.array(hand).flatten().tolist()])

    def close(self):
        self.csv_file.close()


class HandLandmarksDetector():
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5)

    def detectHand(self, frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    hand.extend([x, y, z])
            hands.append(hand)
        return hands, annotated_image


def run(data_path, sign_img_path, split="val", resolution=(1280, 720)):
    hand_detector = HandLandmarksDetector()
    cam = cv2.VideoCapture(0)
    cam.set(3, resolution[0])
    cam.set(4, resolution[1])

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(sign_img_path, exist_ok=True)
    print(sign_img_path)
    dataset_path = f"{data_path}/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)
    current_letter = None
    status_text = None
    cannot_switch_char = False

    saved_frame = None
    while cam.isOpened():
        _, frame = cam.read()
        hands, annotated_image = hand_detector.detectHand(frame)

        if (current_letter is None):
            status_text = "press a character to record"

        else:
            label = ord(current_letter) - ord("a")
            if label == -65:
                status_text = f"Recording unknown, press spacebar again to stop"
                label = -1
            else:
                status_text = f"Recording {LABEL_TAG[label]}, press {current_letter} again to stop"

        key = cv2.waitKey(1)
        if (key == -1):
            if (current_letter is None):
                # no current letter recording, just skip it
                pass
            else:
                if len(hands) != 0:
                    hand = hands[0]
                    hand_dataset.add(hand=hand, label=label)
                    saved_frame = frame
        # some key is pressed
        else:
            # pressed some key, do not push this image, assign current letter to the key just pressed
            key = chr(key)
            if key == "q":
                break
            if (is_handsign_character(key)):
                if (current_letter is None):
                    current_letter = key
                elif (current_letter == key):
                    # pressed again?, reset the current state
                    if saved_frame is not None:
                        if label >= 0:
                            cv2.imwrite(f"{sign_img_path}/{LABEL_TAG[label]}.jpg", saved_frame)

                    cannot_switch_char = False
                    current_letter = None
                    saved_frame = None
                else:
                    cannot_switch_char = True
                    # warned user to unbind the current_letter first
        if (cannot_switch_char):
            cv2.putText(annotated_image, f"please press {current_letter} again to unbind", (0, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_image, status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}", annotated_image)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(data_path, sign_img_path, "train", (1280, 720))
    run(data_path, sign_img_path, "val", (1280, 720))
    run(data_path, sign_img_path, "test", (1280, 720))