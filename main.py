from ultralytics import YOLO
from util import read_license_plate, write_csv
import torch
import cv2

if __name__ == '__main__':
    results = {}
    device = torch.device('cuda:0')
    license_plate_detector = YOLO('./models/bestv1.pt').to(device)
    cap = cv2.VideoCapture('./input/jalan_raya.mp4')
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        # if frame_nmr == 100:
        #     break
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)


                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)


                if license_plate_text is not None:

                    # cv2.imshow("ori", license_plate_crop)
                    # cv2.imshow("new", license_plate_crop_thresh)
                    # print(license_plate_text)
                    # print(license_plate_text_score)
                    # cv2.waitKey(0)

                    results[frame_nmr][0] = {'car': {'bbox': [0, 0, 0, 0]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                       
write_csv(results, './test.csv')