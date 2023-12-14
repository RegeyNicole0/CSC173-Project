# DETECTION AND BLURRING [NOT REAL-TIME]
import cv2
import argparse
from ultralytics import YOLO
import os

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Censor Cam")
    parser.add_argument("--input-type", choices=["video", "image"], default="video", help="Type of input: 'video' or 'image'")
    parser.add_argument("--input-path", required=True, help="Path to the input video or image directory")
    parser.add_argument("--output-path", default="output.mp4", help="Path to the output video or image directory")
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

def process_video(input_path, output_path, webcam_resolution, model):
    cap = cv2.VideoCapture(input_path)
    frame_width, frame_height = webcam_resolution
    out = cv2.VideoWriter(output_path, cv2.CV_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO prediction on the frame
        results = model.predict(frame, conf=0.5)

        # Iterate through the list of results and draw bounding boxes on the frame
        for result in results:
            # Accessing bounding box information using the 'boxes' attribute
            if result.boxes is not None:
                # Call the blur_boxes function to apply Gaussian blur to the regions inside the bounding boxes
                frame = blur_boxes(frame, result.boxes.xyxy)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame with bounding boxes and blurred regions
        cv2.imshow("Object Detector", frame)

        # Check for the 'Esc' key to exit
        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_images(input_path, output_directory, model):
    for filename in os.listdir(input_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_path, filename)
            frame = cv2.imread(img_path)

            # Perform YOLO prediction on the frame
            results = model.predict(frame, conf=0.5)

            # Iterate through the list of results and draw bounding boxes on the frame
            for result in results:
                # Accessing bounding box information using the 'boxes' attribute
                if result.boxes is not None:
                    # Call the blur_boxes function to apply Gaussian blur to the regions inside the bounding boxes
                    frame = blur_boxes(frame, result.boxes.xyxy)

            # Save the processed image to the output directory
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, frame)

            # Display the frame with bounding boxes and blurred regions
            cv2.imshow("Object Detector", frame)

            # Check for the 'Esc' key to exit
            if (cv2.waitKey(0) == 27):
                break

    cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    model = YOLO("best.pt")

    if args.input_type == "video":
        process_video(args.input_path, args.output_path, args.webcam_resolution, model)
    elif args.input_type == "image":
        process_images(args.input_path, args.output_path, model)

if __name__ == "__main__":
    main()
