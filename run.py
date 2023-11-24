import cv2
import argparse
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Censor Cam")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("gunsKnivesDetector.pt")

    while True:
        ret, frame = cap.read()

        # Perform YOLO prediction on the frame
        results = model.predict(frame, conf=0.5)

        # Iterate through the list of results and draw bounding boxes on the frame
        for result in results:
            # Accessing bounding box information using the 'boxes' attribute
            if result.boxes is not None:
                for box in result.boxes.xyxy:
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow("Object Detection", frame)

        # Check for the 'Esc' key to exit
        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    main()
