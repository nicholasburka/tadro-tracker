import numpy as np
import cv2 as cv
import math
from os.path import normpath
import copy

'''
TODO:
Heading info

1. Try to identify Blue and Green LEDs uniquely [x]
2. Store their position [ ]
3. Make a vector towards green from blue [ ]


Add arrows to visualization [ ]
Change visualization to white background [ ]

Convert frame numbering into numbering by timestamp [ ]



TOTHINKABOUT:
Trajectory info with the heading info

Use probabistic method to narrow out blobs

Cluster LED points to (blobs seperated by edge detection) to find the
Tadro blob.

Make color switching more continuous and more colors.

Incorporate skew data so output is absolute, rather than fisheye.
'''
###################### GLOBAL SYSTEM STATE ########################

# class for a generic data holder
class Data:
    def __init__(self): pass    # empty constructor
    
# object (of class Data) to hold the global state of the system
D = Data()


class Tadro:
    def __init__(self, time, position, heading):
        #in either frames or milliseconds
        self.time = time
        self.position = position
        
        #angle with respect to the x axis
        self.heading = heading

#################### YOU NEED TO SET THESE! #######################

D.GREEN = 0
D.BLUE = 1 

#specify the filepath to the video (inside normpath!)
D.VIDEO_PATH = normpath("C:/Users/RoboMaster/Documents/INSPIRE GRANT/Tadro Tracking/GOPR0442.mp4")

#the number of frames to skip to get to the beginning of the tadro's swimming
D.NUM_FRAMES_TO_SKIP = 1000

#the number of frames to skip in between each image processing iteration
D.FRAME_RATE = 120

#whether or not to use preset thresholding values from thresh.txt
D.AUTO_LOAD_THRESHOLDS = True

#do you want to see the Tadro video (Graphical User Interface) as it is processed?
D.USE_GUI = True

#save the data to a file
D.SAVE_POSNS = True

#whether or not to use a video of camera calibration to subtract the pool background
D.CAMERA_CALIBRATION_SUBTRACT = False

#the path to a camera calibration video (should be matched with the supplied Tadro video)
D.CAMERA_CALIBRATION_PATH = normpath("C:/Users/RoboMaster/Documents/INSPIRE GRANT/Tadro Tracking/Camera Calibration.mp4")

D.NUM_CALIBRATION_FRAMES_TO_SKIP = 500

#Use adaptive thresholding to reduce image noise
D.ADAPTIVE_THRESHOLD = False

#this doesn't work
D.BACKGROUND_EXTRACTION = False

#D.START_POS_BOUNDING_BOX = [(

# do you want small windows (about 320 x 240)
D.half_size = False



#################### INITIALIZATION FUNCTIONS ######################

def init_globals():
    """ sets up the data we need in the global dictionary D
    """
    # get D so that we can change values in it
    global D

    D.tadro_data = []


    # put threshold values into D
    D.thresholds =  [{},{}]
    D.thresholds[D.GREEN] =  {'low_red':0, 'high_red':255,
                       'low_green':0, 'high_green':255,
                       'low_blue':0, 'high_blue':255,
                       'low_hue':0, 'high_hue':255,
                       'low_sat':0, 'high_sat':255,
                       'low_val':0, 'high_val':255 }
    D.thresholds[D.BLUE] =  {'low_red':0, 'high_red':255,
                       'low_green':0, 'high_green':255,
                       'low_blue':0, 'high_blue':255,
                       'low_hue':0, 'high_hue':255,
                       'low_sat':0, 'high_sat':255,
                       'low_val':0, 'high_val':255 }

    if (D.USE_GUI):
        # Set up the windows containing the image from the camera,
        # the altered image, and the threshold sliders.
        cv.namedWindow('image')
        cv.moveWindow('image', 0, 0)

        for i in range(len(D.thresholds)):
            print i

            cv.namedWindow('threshold%d' % i)
            THR_WIND_OFFSET = 640
            if D.half_size: THR_WIND_OFFSET /= 2
            cv.moveWindow('threshold%d' % i, THR_WIND_OFFSET, 0)

            cv.namedWindow('sliders%d' % i)
            SLD_WIND_OFFSET = 1280
            if D.half_size: SLD_WIND_OFFSET /= 2
            cv.moveWindow('sliders%d' % i, SLD_WIND_OFFSET, 0)
            cv.resizeWindow('sliders%d' % i,400,600)

        cv.createTrackbar('low_red', 'sliders%d' % 0, D.thresholds[0]['low_red'], 255, 
                              lambda x: change_slider(0, 'low_red', x) )
        cv.createTrackbar('high_red', 'sliders%d' % 0, D.thresholds[0]['high_red'], 255, 
                              lambda x: change_slider(0, 'high_red', x) )
        cv.createTrackbar('low_green', 'sliders%d' % 0, D.thresholds[0]['low_green'], 255, 
                              lambda x: change_slider(0, 'low_green', x) )
        cv.createTrackbar('high_green', 'sliders%d' % 0, D.thresholds[0]['high_green'], 255, 
                              lambda x: change_slider(0, 'high_green', x) )
        cv.createTrackbar('low_blue', 'sliders%d' % 0, D.thresholds[0]['low_blue'], 255, 
                              lambda x: change_slider(0,'low_blue', x) )
        cv.createTrackbar('high_blue', 'sliders%d' % 0, D.thresholds[0]['high_blue'], 255, 
                              lambda x: change_slider(0,'high_blue', x) )
        cv.createTrackbar('low_sat', 'sliders%d' % 0, D.thresholds[0]['low_sat'], 255, 
                              lambda x: change_slider(0,'low_sat', x))
        cv.createTrackbar('high_sat', 'sliders%d' % 0, D.thresholds[0]['high_sat'], 255, 
                              lambda x: change_slider(0,'high_sat', x))
        cv.createTrackbar('low_hue', 'sliders%d' % 0, D.thresholds[0]['low_hue'], 255, 
                              lambda x: change_slider(0,'low_hue', x))
        cv.createTrackbar('high_hue', 'sliders%d' % 0, D.thresholds[0]['high_hue'], 255, 
                              lambda x: change_slider(0,'high_hue', x))
        cv.createTrackbar('low_val', 'sliders%d' % 0, D.thresholds[0]['low_val'], 255, 
                              lambda x: change_slider(0,'low_val', x))
        cv.createTrackbar('high_val', 'sliders%d' % 0, D.thresholds[0]['high_val'], 255, 
                              lambda x: change_slider(0,'high_val', x))

        cv.createTrackbar('low_red', 'sliders%d' % 1, D.thresholds[1]['low_red'], 255, 
                              lambda x: change_slider(1, 'low_red', x) )
        cv.createTrackbar('high_red', 'sliders%d' % 1, D.thresholds[1]['high_red'], 255, 
                              lambda x: change_slider(1, 'high_red', x) )
        cv.createTrackbar('low_green', 'sliders%d' % 1, D.thresholds[1]['low_green'], 255, 
                              lambda x: change_slider(1, 'low_green', x) )
        cv.createTrackbar('high_green', 'sliders%d' % 1, D.thresholds[1]['high_green'], 255, 
                              lambda x: change_slider(1, 'high_green', x) )
        cv.createTrackbar('low_blue', 'sliders%d' % 1, D.thresholds[1]['low_blue'], 255, 
                              lambda x: change_slider(1,'low_blue', x) )
        cv.createTrackbar('high_blue', 'sliders%d' % 1, D.thresholds[1]['high_blue'], 255, 
                              lambda x: change_slider(1,'high_blue', x) )
        cv.createTrackbar('low_sat', 'sliders%d' % 1, D.thresholds[1]['low_sat'], 255, 
                              lambda x: change_slider(1,'low_sat', x))
        cv.createTrackbar('high_sat', 'sliders%d' % 1, D.thresholds[1]['high_sat'], 255, 
                              lambda x: change_slider(1,'high_sat', x))
        cv.createTrackbar('low_hue', 'sliders%d' % 1, D.thresholds[1]['low_hue'], 255, 
                              lambda x: change_slider(1,'low_hue', x))
        cv.createTrackbar('high_hue', 'sliders%d' % 1, D.thresholds[1]['high_hue'], 255, 
                              lambda x: change_slider(1,'high_hue', x))
        cv.createTrackbar('low_val', 'sliders%d' % 1, D.thresholds[1]['low_val'], 255, 
                              lambda x: change_slider(1,'low_val', x))
        cv.createTrackbar('high_val', 'sliders%d' % 1, D.thresholds[1]['high_val'], 255, 
                              lambda x: change_slider(1,'high_val', x))


    else:
        cv.namedWindow('buttonPresses')
    # Set the method to handle mouse button presses
    cv.setMouseCallback('image', onMouse, None)

    # We have not created our "scratchwork" images yet
    D.created_images = False

    # Variable for key presses
    D.last_key_pressed = 255

    D.last_posn = (0,0)
    D.velocity = 40
        


def init_images():
    """ Creates all the images we'll need. Is separate from init_globals 
        since we need to know what size the images are before we can make
        them
    """
    # get D so that we can change values in it
    global D

    # Find the size of the image 
    # We set D.image right before calling this function
    D.size = D.image.shape

    #print D.size
    # Create images for each color channel
    D.red = np.zeros(D.size)
    D.blue = np.zeros(D.size)
    D.green = np.zeros(D.size)
    D.hue = np.zeros(D.size)
    D.sat = np.zeros(D.size)
    D.val = np.zeros(D.size)

    # Create images to save the thresholded images to
    
    D.red_threshed = np.eye(*D.size)
    D.green_threshed = np.eye(*D.size)
    D.blue_threshed = np.eye(*D.size)
    D.hue_threshed = np.eye(*D.size)
    D.sat_threshed = np.eye(*D.size)
    D.val_threshed = np.eye(*D.size)
    

    # The final thresholded result
    D.threshed_images = [np.eye(1), np.eye(1)]
    D.threshed_images[D.GREEN] = np.eye(*D.size)
    D.threshed_images[D.BLUE] = np.eye(*D.size)

    # Create an hsv image and a copy for contour-finding
    D.hsv = np.eye(*D.size)
    D.copy = np.eye(*D.size)
    #D.storage = cv.CreateMemStorage(0) # Create memory storage for contours

    # bunch of keypress values
    # So we know what to show, depending on which key is pressed
    D.key_dictionary = {ord('w'): D.threshed_images,
                        ord('u'): D.red,
                        ord('i'): D.green,
                        ord('o'): D.blue,
                        ord('j'): D.red_threshed,
                        ord('k'): D.green_threshed,
                        ord('l'): D.blue_threshed,
                        ord('a'): D.hue,
                        ord('s'): D.sat,
                        ord('d'): D.val,
                        ord('z'): D.hue_threshed,
                        ord('x'): D.sat_threshed,
                        ord('c'): D.val_threshed,
                        }

    # set the default image for the second window
    D.current_threshold = D.threshed_images

    # Obtain the image from the camera calibration to subtract from the captured image
    if(D.CAMERA_CALIBRATION_SUBTRACT):
        cap = cv.VideoCapture(D.CAMERA_CALIBRATION_PATH)
        if(not cap.isOpened()):
            raise NameError("Invalid camera calibration file path. Turn off camera calibration subtraction or correct.")
        else:
            print "Camera calibration path exists."
        for i in range(0, D.NUM_CALIBRATION_FRAMES_TO_SKIP):
            cap.read()
        ret, frame = cap.read()
        D.calibration_image = frame

def load_thresholds(path="./thresh.txt"):
    # should check if file exists!
    f = open(path, "r" ) # open the file "thresh.txt" for reading
    data = f.read() # read everything from f into data
    x = eval( data ) # eval is Python's evaluation function
    # eval evaluates strings as if they were at the Python shell
    f.close() # its good to close the file afterwards
    print "(b) Loaded thresholds from thresh.txt. Use 'v' to save them."

    # Set threshold values in D
    D.thresholds = x

    if (D.USE_GUI):
        # Update threshold values on actual sliders
        for j in range(len(D.thresholds)):
            for i, x in enumerate(['low_red', 'high_red', 'low_green', 'high_green', 'low_blue', 'high_blue',
                          'low_hue', 'high_hue', 'low_sat', 'high_sat', 'low_val', 'high_val']):
                cv.setTrackbarPos(x + str(j), 'sliders%d' % j, D.thresholds[j][x])

def make_tadro_path_image():

    #makes the output image produce RGBA (A for Alpha, allowing for transparent pixels)
    #instead of just RBG like the input image. 4 channels instead of three
    D.tadro_image_size = (D.size[0], D.size[1], 4)
    D.tadro_image = np.zeros(D.tadro_image_size)
    col = np.array([0,0,0])
    counter = 0
    for i, x in enumerate(D.tadro_data):
        if (x[1] == None):
            continue
        '''
        if (counter == 0):
            col = np.array([255, 255, x[0]%256], copy=True)
        elif(counter == 1):
            col = np.array([x[0]%256, 255, 255], copy=True)
        elif(counter == 2):
            col = np.array([255, x[0]%256, 255], copy=True)
            
        if (x[0]%256 == 0):
                counter += 1
                counter = counter%3
        '''

        if (counter == 0):
            col = np.array([0, 0, i%256, 255], copy=True)
        elif(counter == 1):
            col = np.array([i%256, 0, 0, 255], copy=True)
        elif(counter == 2):
            col = np.array([0, i%256, 0, 255], copy=True)
            
        if (i%256 == 0):
                counter += 1
                counter = counter%3

                
        #print col
        cv.circle(D.tadro_image, x[1], 1, copy.copy(col))
    cv.imshow('threshold0', D.tadro_image)

def make_tadro_path_heading_image():
    D.tadro_image_size = (D.size[0], D.size[1], 4)
    D.tadro_image = np.zeros(D.tadro_image_size)
    col = np.array([0,0,0, 255])
    counter = 0
    for i, x in enumerate(D.tadro_data):
        if (x[1] == None):
            continue

        if (counter == 0):
            col = np.array([0, 0, i%256, 255], copy=True)
        elif(counter == 1):
            col = np.array([i%256, 0, 0, 255], copy=True)
        elif(counter == 2):
            col = np.array([0, i%256, 0, 255], copy=True)
            
        if (i%256 == 0):
                counter += 1
                counter = counter%3

        back = x[3]
        front = x[2]
        center_line = (back, front)
        #angle of arrow in radians
        arrow_angle = .3

        #rotating the back LED about the front LED to make an arrow
        right_shift_back_x = int(front[0] + (back[0] - front[0])*math.cos(arrow_angle) - (back[1] - front[1])*math.sin(arrow_angle))
        right_shift_back_y = int(front[1] + (back[1] - front[1])*math.cos(arrow_angle) - (back[0] - front[0])*math.sin(arrow_angle))

        left_shift_back_x = int(front[0] + (back[0] - front[0])*math.cos(-1*arrow_angle) - (back[1] - front[1])*math.sin(-1*arrow_angle))
        left_shift_back_y = int(front[1] + (back[1] - front[1])*math.cos(-1*arrow_angle) - (back[0] - front[0])*math.sin(-1*arrow_angle))

        cv.line(D.tadro_image, back, front, col, 2)
        cv.circle(D.tadro_image, front, 3, np.array([255,0,0,255]))
        #cv.line(D.tadro_image, (right_shift_back_x, right_shift_back_y), front, col, 2)
        #cv.line(D.tadro_image, (left_shift_back_x, left_shift_back_y), front, col, 2)



def extract_background():
    global D

    cap = cv.VideoCapture(D.VIDEO_PATH)
    D.background_extractor = cv.BackgroundSubtractorMOG()
    for i in range(0, D.NUM_FRAMES_TO_SKIP):
        cap.read()
    while (cap.isOpened() and frame != None):
        ret, frame = cap.read()
        D.background_extractor.apply(frame)
    
################## END INITIALIZATION FUNCTIONS ####################
    

 ################### IMAGE PROCESSING FUNCTIONS #####################

def threshold_image():
    """ runs the image processing in order to create a 
        black and white thresholded image out of D.image
        into D.threshed_images.
    """
    # get D so that we can change values in it
    global D


    if(D.CAMERA_CALIBRATION_SUBTRACT):
        D.image = cv.subtract(D.image, D.calibration_image)

    if(D.ADAPTIVE_THRESHOLD):
        D.grey = cv.cvtColor(D.image, cv.COLOR_RGB2GRAY)
        #D.grey = np.array(D.grey, np.int32)
        D.adaptive_thresh = cv.adaptiveThreshold(D.grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        


    # D.image.shape[2] gives the number of channels
    # Use OpenCV to split the image up into channels, saving them in gray images
    D.BGRchannels = cv.split(D.image)
    #print D.BGRchannels
    D.blue = D.BGRchannels[0]
    D.green = D.BGRchannels[1]
    D.red = D.BGRchannels[2]

    # This line creates a hue-saturation-value image
    D.hsv = cv.cvtColor(D.image, cv.COLOR_BGR2HSV)
    #print D.image.shape
    #print D.hsv
    #print D.hsv.shape
    #print cv.split(D.hsv)
    D.HSVchannels = cv.split(D.hsv)
    #print D.HSVchannels
    D.hue = D.HSVchannels[0]
    D.sat = D.HSVchannels[1]
    D.val = D.HSVchannels[2]

    for i in range(len(D.thresholds)):
         # Here is how OpenCV thresholds the images based on the slider values:
        D.red_threshed = np.eye(*D.size)
        D.blue_threshed = np.eye(*D.size)
        D.green_threshed = np.eye(*D.size)
        D.hue_threshed = np.eye(*D.size)
        D.sat_threshed = np.eye(*D.size)
        D.val_threshed = np.eye(*D.size)

        # Multiply all the thresholded images into one "output" image, D.threshed_images
        D.threshed_images[i] = np.eye(*D.size)

        # Here is how OpenCV thresholds the images based on the slider values:
        D.red_threshed = cv.inRange(D.red, D.thresholds[i]["low_red"], D.thresholds[i]["high_red"], D.red_threshed)
        D.blue_threshed = cv.inRange(D.blue, D.thresholds[i]["low_blue"], D.thresholds[i]["high_blue"], D.blue_threshed)
        D.green_threshed = cv.inRange(D.green, D.thresholds[i]["low_green"], D.thresholds[i]["high_green"], D.green_threshed)
        D.hue_threshed = cv.inRange(D.hue, D.thresholds[i]["low_hue"], D.thresholds[i]["high_hue"], D.hue_threshed)
        D.sat_threshed = cv.inRange(D.sat, D.thresholds[i]["low_sat"], D.thresholds[i]["high_sat"], D.sat_threshed)
        D.val_threshed = cv.inRange(D.val, D.thresholds[i]["low_val"], D.thresholds[i]["high_val"], D.val_threshed)

        # Multiply all the thresholded images into one "output" image, D.threshed_images
        D.threshed_images[i] = cv.multiply(D.red_threshed, D.green_threshed, D.threshed_images[i])
        D.threshed_images[i] = cv.multiply(D.threshed_images[i], D.blue_threshed, D.threshed_images[i])
        D.threshed_images[i] = cv.multiply(D.threshed_images[i], D.hue_threshed, D.threshed_images[i])
        D.threshed_images[i] = cv.multiply(D.threshed_images[i], D.sat_threshed, D.threshed_images[i])
        D.threshed_images[i] = cv.multiply(D.threshed_images[i], D.val_threshed, D.threshed_images[i])

        if(D.ADAPTIVE_THRESHOLD):
            D.threshed_images[i] = cv.multiply(D.threshed_images[i], D.adaptive_thresh, D.threshed_images[i])

    #D.threshed_images = cv.dilate(D.threshed_images, None, iterations=2)

    #cv.imshow(D.threshed_images)
    # Erode and Dilate shave off and add edge pixels respectively
    #cv.Erode(D.threshed_images, D.threshed_images, iterations = 1)
    #cv.Dilate(D.threshed_images, D.threshed_images, iterations = 1)

def are_these_leds(x1, y1, x2, y2):
    #to be improved later, this is just a simple heuristic that says the LEDs will
    #by definition of their proximity be within a certain distance of each other

    #later, it would be nice to know exactly how far apart they should be based on the skew grid
    #and build a stronger heuristic from that
    MAX_DIST = 500
    MIN_DIST = 1
    dist = math.sqrt(abs(int(x2) - int(x1))**2 + abs(int(y2) - int(y1))**2)

    answer = MIN_DIST < dist < MAX_DIST

    return answer
    


def find_leds():
    """ finds all the contours in threshed image, finds the largest of those,
        and then marks in in the main image
    """
    # get D so that we can change values in it
    global D

    # initialize list of LED posns to len of thresholds
    LEDs = [0 for k in range(len(D.thresholds))]

    for i in range(len(D.threshed_images)):
        # Create a copy image of thresholds then find contours on that image
        D.copy = D.threshed_images[i].copy() # copy threshed image

        

        # this is OpenCV's call to find all of the contours:
        _, contours, _ = cv.findContours(D.copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Next we want to find the *largest* contour
        # this is the standard algorithm:
        #    walk the list of all contours, remembering the biggest so far:
        if len(contours) > 0:
            biggest = contours[0]
            second_biggest = contours[0]
            biggestArea = cv.contourArea(contours[0]) #get first contour
            secondArea = cv.contourArea(contours[0])
            for x in contours:
                nextArea = cv.contourArea(x)
                if biggestArea < nextArea:
                    second_biggest = biggest
                    biggest = x
                    secondArea = biggestArea
                    biggestArea = nextArea

            
            # Use OpenCV to get a bounding rectangle for the largest contour
            br = cv.boundingRect(biggest)

            # Make a bounding box around the biggest blob
            upper_left = (br[0], br[1])
            lower_left = (br[0], br[1] + br[3])
            lower_right = (br[0] + br[2], br[1] + br[3])
            upper_right = (br[0] + br[2], br[1])
            cv.polylines(D.image, [np.array([upper_left,lower_left,lower_right,upper_right], dtype=np.int32)],
                        1, np.array([255, 0, 0]))
            cv.polylines(D.threshed_images[i], [np.array([upper_left,lower_left,lower_right,upper_right], dtype=np.int32)],
                        1, np.array([255, 0, 0]))

            #Store the contour info for the biggest blob, which we assume is the LED based on thresholding
            LEDs[i] = biggest

    #print biggest
    #print second_biggest
    #calculate moments for biggest and second biggest blobs
    moment0 = cv.moments(LEDs[0])
    moment1 = cv.moments(LEDs[1])

    if (moment0['m00'] > 0):
        center_x = moment0['m10']/moment0['m00']
        center_y = moment0['m01']/moment0['m00']
        D.blue_pos = (int(center_x), int(center_y))
    else:
        D.blue_pos = None

    if (moment1['m00'] > 0):
        second_center_x = moment1['m10']/moment1['m00']
        second_center_y = moment1['m01']/moment1['m00']
        D.green_pos = (int(second_center_x), int(second_center_y))
    else:
        D.green_pos = None


    #if these blobs have areas > 0, then calculate the average of their centroids
    if (moment0['m00'] > 0 and moment1['m00'] > 0):

        led_check = are_these_leds(center_x, center_y, second_center_x, second_center_y)

        if (led_check):
            D.tadro_center = (int((center_x + second_center_x)/2), int((center_y + second_center_y)/2))
            cv.circle(D.image, D.tadro_center, 10, np.array([255, 255, 0]))
            cv.circle(D.threshed_images[0], D.tadro_center, 10, np.array([255, 255, 0]))
        else:
            D.tadro_center = None
        
    #else simply calculate the centroid of the largest blob
    else:
        D.tadro_center = None

    # Draw matching contours in white with inner ones in green
    # cv.DrawContours(D.image, biggest, cv.RGB(255, 255, 255), 
    #               cv.RGB(0, 255, 0), 1, thickness=2, lineType=8, 
    #               offset=(0,0))


################# END IMAGE PROCESSING FUNCTIONS ###################

                

####################### CALLBACK FUNCTIONS #########################

def onMouse(event, x, y, flags, param):
    """ the method called when the mouse is clicked """
    global D
    
    # clicked the left button
    if event==cv.EVENT_LBUTTONDOWN: 
        print "x, y are", x, y, "    ",
        (b,g,r) = D.image[y,x]
        print "r,g,b is", int(r), int(g), int(b), "    ",
        (h,s,v) = D.hsv[y,x]
        print "h,s,v is", int(h), int(s), int(v)
        D.down_coord = (x,y)



def check_key_press(key_press):
    """ this handler is called when a real key press has been
        detected, and updates everything appropriately
    """
    # get D so that we can change values in it
    global D
    D.last_key_pressed = key_press

    # if it was ESC, make it 'q'
    if key_press == 27:
        key_press = ord('q')

    # if a 'q' or ESC was pressed, we quit
    if key_press == ord('q'): 
        print "quitting"
        return

    # help menu
    if key_press == ord('h'):
        print " Keyboard Command Menu"
        print " =============================="
        print " q    : quit"
        print " ESC  : quit"
        print " h    : help menu"
        print " w    : show total threshold image in threshold window"
        print " r    : show red image in threshold window"
        print " t    : show green image in threshold window"
        print " y    : show blue image in threshold window"
        print " f    : show thresholded red image in threshold window"
        print " g    : show thresholded blue image in threshold window"
        print " h    : show thresholded green image in threshold window"
        print " a    : show hue image in threshold window"
        print " s    : show saturation image in threshold window"
        print " d    : show value image in threshold window"
        print " z    : show thresholded hue image in threshold window"
        print " x    : show thresholded saturation image in threshold window"
        print " c    : show thresholded value image in threshold window"
        print " v    : saves threshold values to file (overwriting)"
        print " b    : loads threshold values from file"
        print " u    : mousedrags no longer set thresholds"
        print " i    : mousedrag set thresholds to area within drag"

    elif key_press == ord('v'):
        x = D.thresholds
        f = open( "./thresh.txt", "w" ) # open the file "thresh.txt" for writing
        print >> f, x # print x to the file object f
        f.close() # it's good to close the file afterwards
        print "(v) Wrote thresholds to thresh.txt. Use 'b' to load them."

    elif key_press == ord('b'):
        load_thresholds()
    elif key_press == ord('s'):
        print "saving position data to posns.txt..."
        x = D.tadro_data
        f = open( "./posns.txt", "w" ) # open the file "thresh.txt" for writing
        print >> f, x # print x to the file object f
        f.close() # it's good to close the file afterwards
        print "save complete."

    # threshold keypresses:
    elif key_press in D.key_dictionary.keys():
        D.current_threshold = D.key_dictionary[key_press]


# Function for changing the slider values
def change_slider(i, name, new_threshold):
    """ a small function to change a slider value """
    # get D so that we can change values in it
    global D
    print name
    D.thresholds[i][name] = new_threshold


#get image, threshold, and analyze for Tadro
def handle_image():
    """ this function organizes all of the processing
        done for each image from a camera or Kinect
    """
    # get D so that we can change values in it
    global D

    if D.orig_image == None: # did we get an image at all?
        print "No image"
        return

    D.image = D.orig_image

    if D.created_images == False:   # have we set up the other images?
        init_images()               # Initialize the others needed
        D.created_images = True     # We only need to run this one time

    # Recalculate threshold image
    threshold_image()

    find_leds()
    #find_tadro()

    # Get any incoming keypresses
    # To get input from keyboard, we use cv.WaitKey
    # Only the lowest eight bits matter (so we get rid of the rest):
    key_press_raw = cv.waitKey(5) # gets a raw key press
    key_press = key_press_raw & 255 # sets all but the low 8 bits to 0
    
    # Handle key presses only if it's a real key (255 = "no key pressed")
    if key_press != 255:
        check_key_press(key_press)

    if (D.USE_GUI):
        # Update the displays:
        # Main image:
        cv.imshow('image', D.image)

        # Currently selected threshold image:
        for i in range(len(D.threshed_images)):
            cv.imshow('threshold%d' % i, D.threshed_images[i])#D.current_threshold )



def handle_kinect_data(data):
    """ this function grabs images from the Kinect
    """
    # get D so that we can change values in it
    global D

    # Get the incoming image from the Kinect
    D.orig_image = D.bridge.imgmsg_to_cv(data, "bgr8")

    # now, handle that image...
    handle_image()

##################### END CALLBACK FUNCTIONS #######################

        

############################## MAIN ################################

def main():
    """ the main program that sets everything up
    """
    global D

    # Initialize all the global variables we will need
    init_globals()
    print "working..."

    #get the video file
    cap = cv.VideoCapture(D.VIDEO_PATH)

    #make sure the video file is valid
    print D.VIDEO_PATH + " is an accessible video filepath?"
    if (not cap.isOpened()):
        raise NameError("Invalid video filepath.")
    else:
        print "TRUE"

    if (D.BACKGROUND_EXTRACTION):
        extract_background()
    
    if (D.AUTO_LOAD_THRESHOLDS):
        load_thresholds()

    #skip D.NUM_FRAMES_TO_SKIP
    for i in range(0, D.NUM_FRAMES_TO_SKIP):
        cap.grab()
    j = D.NUM_FRAMES_TO_SKIP
    
    while(cap.isOpened()):
        #get the current frame
        ret, frame = cap.read()

        if (D.BACKGROUND_EXTRACTION):
            frame = D.background_extractor.apply(frame)

        #process the image
        D.orig_image = frame
        handle_image()

        #store the data
        D.tadro_data.append((j, D.tadro_center, D.blue_pos, D.green_pos))

        #quit if told to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        #skip FRAME_RATE number of frames
        for i in range(0, D.FRAME_RATE):
            cap.grab()

        #increment the frame counter
        j += 1 + D.FRAME_RATE

    make_tadro_path_heading_image() #make an image displaying Tadro path

    #save the image in the current directory
    cv.imwrite("./path_image.png", D.tadro_image)

    #clean up
    cap.release()
    cv.destroyAllWindows()

# this is the "main" trick - it tells Python
# what to run as a stand-alone script:
if __name__ == "__main__":
    main()
