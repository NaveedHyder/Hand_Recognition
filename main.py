import cv2
import mediapipe as mp
vid =cv2.VideoCapture(0)
#vid.set(3,960)
mphands=mp.solutions.hands
Hands = mphands.Hands(max_num_hands= 1, min_detection_confidence= 0.7, min_tracking_confidence= 0.6 )
mpDraw=mp.solutions.drawing_utils
while True:
    success,frame = vid.read()
    #convert from BGR to RGB
    RGBframe=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=Hands.process(RGBframe)
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            #print(handLm)
            for id, lm in enumerate(handLm.landmark):
                h,w,_=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                cv2.circle(frame,(cx,cy),5,(0,255,0),cv2.FILLED)
                mpDraw.draw_landmarks(frame,handLm,mphands.HAND_CONNECTIONS)
        #print("Hand Found")
    #else:
        #print("Hand not found")    
    #cv2.imshow("RGBVideo",RGBframe)
    cv2.imshow("Video",frame)
    cv2.waitKey(1)
