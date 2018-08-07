# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
	#help="path to facial landmark predictor")
#ap.add_argument("-r", "--picamera", type=int, default=-1,
	#help="whether or not the Raspberry Pi camera should be used")
#args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		
		shape = face_utils.shape_to_np(shape)
		#print(shape)
		

		

		cv2.line(frame, (tuple(shape[17].ravel())), (tuple(shape[18].ravel())), (0, 0, 0), 3) 
		cv2.line(frame, (tuple(shape[18].ravel())), (tuple(shape[19].ravel())), (0, 0, 0), 3)     
		cv2.line(frame, (tuple(shape[19].ravel())), (tuple(shape[20].ravel())), (0, 0, 0), 3)
		cv2.line(frame, (tuple(shape[20].ravel())), (tuple(shape[21].ravel())), (0, 0, 0), 3)

		cv2.line(frame, (tuple(shape[36].ravel())), (tuple(shape[37].ravel())), (0, 0, 0), 3) 
		cv2.line(frame, (tuple(shape[37].ravel())), (tuple(shape[38].ravel())), (0, 0, 0), 3)     
		cv2.line(frame, (tuple(shape[38].ravel())), (tuple(shape[39].ravel())), (0, 0, 0), 3)

		cv2.line(frame, (tuple(shape[42].ravel())), (tuple(shape[43].ravel())), (0, 0, 0), 3) 
		cv2.line(frame, (tuple(shape[43].ravel())), (tuple(shape[44].ravel())), (0, 0, 0), 3)     
		cv2.line(frame, (tuple(shape[44].ravel())), (tuple(shape[45].ravel())), (0, 0, 0), 3)
	    

		cv2.line(frame, (tuple(shape[22].ravel())), (tuple(shape[23].ravel())), (0, 0, 0), 3)   
		cv2.line(frame, (tuple(shape[23].ravel())), (tuple(shape[24].ravel())), (0, 0, 0), 3)
		cv2.line(frame, (tuple(shape[24].ravel())), (tuple(shape[25].ravel())), (0, 0, 0), 3)  
		cv2.line(frame, (tuple(shape[25].ravel())), (tuple(shape[26].ravel())), (0, 0, 0), 3)
 		
 		#lips
		cv2.line(frame, (tuple(shape[48].ravel())), (tuple(shape[49].ravel())), (128, 0, 32), 3)   
		cv2.line(frame, (tuple(shape[49].ravel())), (tuple(shape[50].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[50].ravel())), (tuple(shape[51].ravel())), (128, 0, 32), 3)  
		cv2.line(frame, (tuple(shape[51].ravel())), (tuple(shape[52].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[52].ravel())), (tuple(shape[53].ravel())), (128, 0, 32), 3)   
		cv2.line(frame, (tuple(shape[54].ravel())), (tuple(shape[55].ravel())), (128, 0, 32), 3)  
		cv2.line(frame, (tuple(shape[53].ravel())), (tuple(shape[54].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[54].ravel())), (tuple(shape[55].ravel())), (128, 0, 32), 3)  
		cv2.line(frame, (tuple(shape[55].ravel())), (tuple(shape[56].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[56].ravel())), (tuple(shape[57].ravel())), (128, 0, 32), 3)   
		cv2.line(frame, (tuple(shape[57].ravel())), (tuple(shape[58].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[58].ravel())), (tuple(shape[59].ravel())), (128, 0, 32), 3)  
		cv2.line(frame, (tuple(shape[59].ravel())), (tuple(shape[48].ravel())), (128, 0, 32), 3) 
		cv2.line(frame, (tuple(shape[60].ravel())), (tuple(shape[61].ravel())), (128, 0, 32), 3)   
		cv2.line(frame, (tuple(shape[60].ravel())), (tuple(shape[61].ravel())), (128, 0, 32), 3)   
		cv2.line(frame, (tuple(shape[61].ravel())), (tuple(shape[62].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[62].ravel())), (tuple(shape[63].ravel())), (128, 0, 32), 3)  
		cv2.line(frame, (tuple(shape[63].ravel())), (tuple(shape[64].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[64].ravel())), (tuple(shape[65].ravel())), (128, 0, 32), 3)   
		cv2.line(frame, (tuple(shape[65].ravel())), (tuple(shape[66].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[66].ravel())), (tuple(shape[67].ravel())), (128, 0, 32), 3)
		cv2.line(frame, (tuple(shape[67].ravel())), (tuple(shape[60].ravel())), (128, 0, 32), 3)


		#rightcheek
		x, y = tuple(shape[15].ravel())
		x1,y1 =tuple(shape[35].ravel())
		midpoint = ((x+x1)/2),((y+y1)/2)
		cv2.circle(frame,midpoint,1,(0,0,255),30)
        
		#leftcheeck
		a, b = tuple(shape[1].ravel())
		a1,b1 =tuple(shape[31].ravel())
		midpoint1 = ((a+a1)/2),((b+b1)/2)
		cv2.circle(frame,midpoint1,1,(0,0,255),30)

		#lefteyebrow
		#firstmidpoint
		c, d = tuple(shape[21].ravel())
		c1,d1 =tuple(shape[40].ravel())
		midpoint2 = ((c+c1)/2), ((d+d1)/2)
		cv2.circle(frame,midpoint2,1,(0,0,255),3)
		e, f = midpoint2
		e1,f1 = tuple(shape[21].ravel())
		midpoint3 = ((e+e1)/2), ((f+f1)/2)
		#secondmidpoint
		k, l = tuple(shape[20].ravel())
		k1,l1 =tuple(shape[39].ravel())
		midpoint6 = ((k+k1)/2), ((l+l1)/2)
		cv2.circle(frame,midpoint6,1,(0,0,255),3)
		a2, b2 = midpoint6
		a3,b3 = tuple(shape[20].ravel())
		midpoint7 = ((a2+a3)/2), ((b2+b3)/2)
		#thirdmidpoint
		a4, b4 = tuple(shape[19].ravel())
		a5, b5 = tuple(shape[38].ravel())
		midpoint8 = ((a4+a5)/2), ((b4+b5)/2)
		cv2.circle(frame,midpoint8,1,(0,0,255),3)
		a6, b6 = midpoint8
		a7, b7 = tuple(shape[19].ravel())
		midpoint9 = ((a6+a7)/2), ((b6+b7)/2)
		#fourthmidpoint
		a8, b8 = tuple(shape[18].ravel())
		a9, b9 = tuple(shape[38].ravel())
		midpoint10 = ((a8+a9)/2), ((b8+b9)/2)
		cv2.circle(frame,midpoint10,1,(0,0,255),3)
		a10, b10 = midpoint10
		a11, b11 = tuple(shape[18].ravel())
		midpoint11 = ((a10+a11)/2), ((b10+b11)/2)
		cv2.line(frame, (tuple(shape[21].ravel())),(midpoint3), (0, 0, 0), 3)
        cv2.line(frame,(midpoint3), (midpoint7), (0, 0, 0), 3)
        cv2.line(frame,(midpoint7), (midpoint9), (0, 0, 0), 3)
        cv2.line(frame,(midpoint9), (midpoint11), (0, 0, 0), 3)
        cv2.line(frame,(midpoint11), (tuple(shape[17].ravel())), (0, 0, 0), 3)

		#righteyebrow
		#firstmidpoint
        g, h = tuple(shape[22].ravel())
        g1,h1 =tuple(shape[43].ravel())
        midpoint4 = ((g+g1)/2), ((h+h1)/2)
        cv2.circle(frame,midpoint4,1,(0,0,255),3)
        i, j = midpoint4
        i1,j1 = tuple(shape[22].ravel())
        midpoint5 = ((i+i1)/2), ((j+j1)/2)
        #secondmidpoint
        c2, d2 = tuple(shape[23].ravel())
        c3,d3 =tuple(shape[44].ravel())
        midpoint13 = ((c2+c3)/2), ((d2+d3)/2)
        cv2.circle(frame,midpoint13,1,(0,0,255),3)
        c4, d4 = midpoint13
        c5,d5 = tuple(shape[23].ravel())
        midpoint14 = ((c4+c5)/2), ((d4+d5)/2)
        #thirdmidpoint
        c6, d6 = tuple(shape[24].ravel())
        c7,d7 =tuple(shape[45].ravel())
        midpoint15 = ((c6+c7)/2), ((d6+d7)/2)
        cv2.circle(frame,midpoint15,1,(0,0,255),3)
        c8, d8 = midpoint15
        c9,d9 = tuple(shape[24].ravel())
        midpoint16 = ((c8+c9)/2), ((d8+d9)/2)
        #fourthmidpoint
        c10, d10 = tuple(shape[25].ravel())
        c11,d11 =tuple(shape[45].ravel())
        midpoint17 = ((c10+c11)/2), ((d10+d11)/2)
        cv2.circle(frame,midpoint17,1,(0,0,255),3)
        c12, d12 = midpoint17
        c13,d13 = tuple(shape[25].ravel())
        midpoint18 = ((c12+c13)/2), ((d12+d13)/2)
        cv2.line(frame, (tuple(shape[22].ravel())),(midpoint5), (0, 0, 0), 3)
        cv2.line(frame,(midpoint5), (midpoint14), (0, 0, 0), 3)
        cv2.line(frame,(midpoint14), (midpoint16), (0, 0, 0), 3)
        cv2.line(frame,(midpoint16), (midpoint18), (0, 0, 0), 3)
        cv2.line(frame,(midpoint18), (tuple(shape[26].ravel())), (0, 0, 0), 3)
        for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		  
	# show the frame
	frame =cv2.flip(frame,1)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 	#time.sleep(60.0)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()