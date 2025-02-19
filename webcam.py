"""
def detect_motion():
    cap = cv2.VideoCapture(0)
    
    # Get the video frame dimensions
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    
    # Create a background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Threshold the mask
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw red rectangles around large contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust this value to change sensitivity
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle

        # Draw a red frame around the entire image
        cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Motion Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_motion()"""
    
#modifiy by advance yolo v3 alg 
import cv2
import numpy as np


def detect_motion():
    cap = cv2.VideoCapture(0)

    # Load YOLO
    net = cv2.dnn.readNet("webcam file\yolov3.weights", "webcam file\yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    classes = []
    with open("webcam file\coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Detecting objects with YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Motion Detection and Object Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('d'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_motion()
