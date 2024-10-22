import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_video(file):
    video = cv2.VideoCapture(file)
    # output = cv2.VideoWriter(f"processed_{file.split('.')[0]}_video.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1280, 720))
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    buf = np.empty((int(total_frames), int(height), int(width), 3), np.dtype('uint8'))

    for fr in range(int(total_frames)):
        ret, buf[fr] = video.read()

    video.release()

    return buf, int(width), int(height)


def write_video(buf, file):
    output = cv2.VideoWriter(f"processed_{file.split('.')[0]}_video.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1280, 720))

    for fr in buf:
        output.write(fr)

    output.release()


def day_night_classifier(buf, width, height):
    brightness_threshold = 100
    overall_frame_bright = 0

    for fr in range(len(buf)):
        frame_area = width * height
        frame_gray = cv2.cvtColor(buf[fr], cv2.COLOR_BGR2GRAY)
        overall_frame_bright += np.sum(frame_gray) / frame_area

    isDay = (overall_frame_bright / len(buf)) >= brightness_threshold

    return isDay


def adjust_brightness(buf):

    for fr in range(len(buf)):

        B, G, R = buf[fr][:,:,0], buf[fr][:,:,1], buf[fr][:,:,2]

        # Equalization
        output1_B = cv2.equalizeHist(B)
        output1_G = cv2.equalizeHist(G)
        output1_R = cv2.equalizeHist(R)

        equ = cv2.merge((output1_B, output1_G, output1_R))
        
        buf[fr] = equ

    return buf


def face_blur(buf):
    face_cascade = cv2.CascadeClassifier("./face_detector.xml")

    for fr in range(len(buf)):
        frame_gray = cv2.cvtColor(buf[fr], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.1, 7)

        for (x,y,w,h) in faces:
            rect = cv2.rectangle(buf[fr], (x, y), (x+w, y+h), (255, 0, 0), 2)
            buf[fr][y:y+h, x:x+w] = cv2.medianBlur(buf[fr][y:y+h, x:x+w], 51)

    return buf
        

def overlay(buf, width, height, fps):
    talk, w, h = read_video("talking.mp4")
    end, w, h = read_video("endscreen.mp4")

    mark1 = cv2.imread("watermark1.png", 0)
    mark2 = cv2.imread("watermark2.png", 0)

    _, mark1_thres = cv2.threshold(mark1, 100, 255, cv2.THRESH_BINARY)
    _, mark2_thres = cv2.threshold(mark2, 100, 255, cv2.THRESH_BINARY)

    mark1_thres = cv2.cvtColor(mark1_thres, cv2.COLOR_GRAY2BGR)
    mark2_thres = cv2.cvtColor(mark2_thres, cv2.COLOR_GRAY2BGR)
    '''
    height * (height * aspectRatio) = area
    height² = area / aspectRatio
    height = sqrt(area / aspectRatio)
    '''
    target_area =  (w * h) * 0.1
    aspectR = w / h
    [new_h] = np.sqrt([target_area/aspectR])
    new_w = new_h * aspectR
    dim = (int(new_w), int(new_h))


    # Overlay
    start = int(height * 0.1)

    for fr in range(len(buf)):
        
        if (fr // (5 * fps)) % 2 == 0:
            buf[fr] = cv2.bitwise_or(buf[fr], mark1_thres)
        else:
            buf[fr] = cv2.bitwise_or(buf[fr], mark2_thres)
        

        if fr < len(talk):
            rs = cv2.copyMakeBorder(cv2.resize(talk[fr], dim), 5, 5, 5, 5, cv2.BORDER_CONSTANT) 
            w, h = len(rs[0]), len(rs)

            buf[fr][start : start + h, start : start + w] = rs

    buf = np.concatenate((buf, end), axis=0)

    return buf


def main():
    videos = ['singapore.mp4']

    for vid in videos:
        buf, w, h = read_video(vid)

        isDay = day_night_classifier(buf, w, h)

        if isDay == False:
            adj = adjust_brightness(buf)
            res = face_blur(adj)
            res = overlay(res, w, h, 30)
            write_video(res, vid)

        else:
            res = face_blur(buf)
            res = overlay(res, w, h, 30)
            write_video(res, vid)


if __name__ == "__main__":
    main()
