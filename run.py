import cv2
import argparse
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Censor Cam")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def blur_boxes(frame, boxes):
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        # Extract the region inside the bounding box
        roi = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        # Apply Gaussian blur to the region
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        # Replace the original region with the blurred one
        frame[int(ymin):int(ymax), int(xmin):int(xmax)] = blurred_roi
    return frame

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
                # Call the blur_boxes function to apply Gaussian blur to the regions inside the bounding boxes
                frame = blur_boxes(frame, result.boxes.xyxy)

        # Display the frame with bounding boxes and blurred regions
        cv2.imshow("Object Detector", frame)

        # Check for the 'Esc' key to exit
        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
