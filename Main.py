import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract' 

def ScanPlate(img):
    # img = cv2.resize(img,(620,480))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(img, 13, 15, 15)
    testvalue = np.mean(gray)
    edged = cv2.Canny(gray, testvalue-21.5, testvalue+21.5)

    contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    screenCnt = 0   

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    print(testvalue)
    mask = np.zeros(gray.shape,np.uint8)                                #masking the rest of image leaving numberplate 
    try:
        numplate = cv2.drawContours(mask,[screenCnt],0,255,-1)
        numplate = cv2.bitwise_and(img,img,mask=mask)
    except:
        print("exception occured")
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    # print(mask)
    # cv2.imshow('numplate',img)
    cv2.imshow('image',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    text = pytesseract.image_to_string(Cropped,config='--psm 11')
    return(text)

path="cars+1.jpg"
img = cv2.imread(path)



# cap = cv2.VideoCapture("E:\Room332\FinalRender.mp4")
# fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count/fps

# print('fps = ' + str(fps))
# print('number of frames = ' + str(frame_count))
# print('duration (S) = ' + str(duration))
# minutes = int(duration/60)
# seconds = duration%60
# print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

# cap.release()


