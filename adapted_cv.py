import numpy as np
import cv2 as cv
import math
from os.path import normpath
import copy

'''
TODO:
Heading info

1. Try to identify Blue and Green LEDs uniquely
2. Store their position
3. Instead of calculating angle, just make a vector towards green from blue


Add arrows to visualization
Change visualization to white background

Convert frame numbering into numbering by timestamp



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

#specify the filepath to the video (inside normpath!)
D.VIDEO_PATH = normpath("C:/Users/RoboMaster/Documents/INSPIRE GRANT/Tadro Tracking/GOPR0442.mp4")

#the number of frames to skip to get to the beginning of the tadro's swimming
D.NUM_FRAMES_TO_SKIP = 1000

#the number of frames to skip in between each image processing iteration
D.FRAME_RATE = 6

#whether or not to use preset thresholding values
D.AUTO_LOAD_THRESHOLDS = True

#do you want to see the Tadro video as it is processed?
D.USE_GUI = False

#save the data to a file
D.SAVE_POSNS = True

#whether or not to use a video of camera calibration to subtract the pool background
D.CAMERA_CALIBRATION_SUBTRACT = False

#the path to a camera calibration video (should be matched with the supplied Tadro video)
D.CAMERA_CALIBRATION_PATH = normpath("C:/Users/RoboMaster/Documents/INSPIRE GRANT/Tadro Tracking/Camera Calibration.mp4")

D.NUM_CALIBRATION_FRAMES_TO_SKIP = 500

#Use adaptive thresholding to reduce image noise
D.ADAPTIVE_THRESHOLD = True

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
    D.thresholds =    {'low_red':0, 'high_red':255,
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

        cv.namedWindow('threshold')
        THR_WIND_OFFSET = 640
        if D.half_size: THR_WIND_OFFSET /= 2
        cv.moveWindow('threshold', THR_WIND_OFFSET, 0)

        cv.namedWindow('sliders')
        SLD_WIND_OFFSET = 1280
        if D.half_size: SLD_WIND_OFFSET /= 2
        cv.moveWindow('sliders', SLD_WIND_OFFSET, 0)
        cv.resizeWindow('sliders',400,600)
    

        # Create the sliders within the 'sliders' window
        cv.createTrackbar('low_red', 'sliders', D.thresholds['low_red'], 255, 
                              lambda x: change_slider('low_red', x) )
        cv.createTrackbar('high_red', 'sliders', D.thresholds['high_red'], 255, 
                              lambda x: change_slider('high_red', x) )
        cv.createTrackbar('low_green', 'sliders', D.thresholds['low_green'], 255, 
                              lambda x: change_slider('low_green', x) )
        cv.createTrackbar('high_green', 'sliders', D.thresholds['high_green'], 255, 
                              lambda x: change_slider('high_green', x) )
        cv.createTrackbar('low_blue', 'sliders', D.thresholds['low_blue'], 255, 
                              lambda x: change_slider('low_blue', x) )
        cv.createTrackbar('high_blue', 'sliders', D.thresholds['high_blue'], 255, 
                              lambda x: change_slider('high_blue', x) )
        cv.createTrackbar('low_sat', 'sliders', D.thresholds['low_sat'], 255, lambda x: change_slider('low_sat', x))
        cv.createTrackbar('high_sat', 'sliders', D.thresholds['high_sat'], 255, lambda x: change_slider('high_sat', x))
        cv.createTrackbar('low_hue', 'sliders', D.thresholds['low_hue'], 255, lambda x: change_slider('low_hue', x))
        cv.createTrackbar('high_hue', 'sliders', D.thresholds['high_hue'], 255, lambda x: change_slider('high_hue', x))
        cv.createTrackbar('low_val', 'sliders', D.thresholds['low_val'], 255, lambda x: change_slider('low_val', x))
        cv.createTrackbar('high_val', 'sliders', D.thresholds['high_val'], 255, lambda x: change_slider('high_val', x))
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
    D.threshed_image = np.eye(*D.size)

    # Create an hsv image and a copy for contour-finding
    D.hsv = np.eye(*D.size)
    D.copy = np.eye(*D.size)
    #D.storage = cv.CreateMemStorage(0) # Create memory storage for contours

    # bunch of keypress values
    # So we know what to show, depending on which key is pressed
    D.key_dictionary = {ord('w'): D.threshed_image,
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
    D.current_threshold = D.threshed_image

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
        for i in ['low_red', 'high_red', 'low_green', 'high_green', 'low_blue', 'high_blue',
                      'low_hue', 'high_hue', 'low_sat', 'high_sat', 'low_val', 'high_val']:
            cv.setTrackbarPos(i, 'sliders', D.thresholds[i])

def make_tadro_path_image():
    D.tadro_image = np.ones(D.size)
    col = np.array([0,0,0])
    counter = 0
    for i, x in enumerate(D.tadro_data):
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
            col = np.array([255, 255, i%256], copy=True)
        elif(counter == 1):
            col = np.array([i%256, 255, 255], copy=True)
        elif(counter == 2):
            col = np.array([255, i%256, 255], copy=True)
            
        if (i%256 == 0):
                counter += 1
                counter = counter%3
                
        #print col
        cv.circle(D.tadro_image, x[1], 1, copy.copy(col))
    cv.imshow('threshold', D.tadro_image)

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
        into D.threshed_image.
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

    # Here is how OpenCV thresholds the images based on the slider values:
    D.red_threshed = cv.inRange(D.red, D.thresholds["low_red"], D.thresholds["high_red"], D.red_threshed)
    D.blue_threshed = cv.inRange(D.blue, D.thresholds["low_blue"], D.thresholds["high_blue"], D.blue_threshed)
    D.green_threshed = cv.inRange(D.green, D.thresholds["low_green"], D.thresholds["high_green"], D.green_threshed)
    D.hue_threshed = cv.inRange(D.hue, D.thresholds["low_hue"], D.thresholds["high_hue"], D.hue_threshed)
    D.sat_threshed = cv.inRange(D.sat, D.thresholds["low_sat"], D.thresholds["high_sat"], D.sat_threshed)
    D.val_threshed = cv.inRange(D.val, D.thresholds["low_val"], D.thresholds["high_val"], D.val_threshed)

    # Multiply all the thresholded images into one "output" image, D.threshed_image
    D.threshed_image = cv.multiply(D.red_threshed, D.green_threshed, D.threshed_image)
    D.threshed_image = cv.multiply(D.threshed_image, D.blue_threshed, D.threshed_image)
    D.threshed_image = cv.multiply(D.threshed_image, D.hue_threshed, D.threshed_image)
    D.threshed_image = cv.multiply(D.threshed_image, D.sat_threshed, D.threshed_image)
    D.threshed_image = cv.multiply(D.threshed_image, D.val_threshed, D.threshed_image)

    if(D.ADAPTIVE_THRESHOLD):
        D.threshed_image = cv.multiply(D.threshed_image, D.adaptive_thresh, D.threshed_image)

    #D.threshed_image = cv.dilate(D.threshed_image, None, iterations=2)

    #cv.imshow(D.threshed_image)
    # Erode and Dilate shave off and add edge pixels respectively
    #cv.Erode(D.threshed_image, D.threshed_image, iterations = 1)
    #cv.Dilate(D.threshed_image, D.threshed_image, iterations = 1)

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
    


def find_biggest_region():
    """ finds all the contours in threshed image, finds the largest of those,
        and then marks in in the main image
    """
    # get D so that we can change values in it
    global D

    # Create a copy image of thresholds then find contours on that image
    D.copy = D.threshed_image.copy() # copy threshed image
    #D.gray_copy = cv.cvtColor(D.copy, cv.COLOR_BGR2GRAY)
    #D.binary_img = cv.threshold(D.gray_copy, 
    

    # this is OpenCV's call to find all of the contours:
    _, contours, _ = cv.findContours(D.copy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print type(contours)
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
                #print biggest
                #print second_biggest
                #print "."
        
        # Use OpenCV to get a bounding rectangle for the largest contour
        br = cv.boundingRect(biggest)

        # print the result of the cv.BoundingRect call...
        #print "br is", br
        #print biggestArea
        #print secondArea

        # Draw a red box from (42,42) to (84,126), for now (you'll change this):
        upper_left = (br[0], br[1])
        lower_left = (br[0], br[1] + br[3])
        lower_right = (br[0] + br[2], br[1] + br[3])
        upper_right = (br[0] + br[2], br[1])
        cv.polylines(D.image, [np.array([upper_left,lower_left,lower_right,upper_right], dtype=np.int32)],
                    1, np.array([255, 0, 0]))
        cv.polylines(D.threshed_image, [np.array([upper_left,lower_left,lower_right,upper_right], dtype=np.int32)],
                    1, np.array([255, 0, 0]))

        br2 = cv.boundingRect(second_biggest)
        upper_left = (br2[0], br2[1])
        lower_left = (br2[0], br2[1] + br2[3])
        lower_right = (br2[0] + br2[2], br2[1] + br2[3])
        upper_right = (br2[0] + br2[2], br2[1])
        cv.polylines(D.image, [np.array([upper_left,lower_left,lower_right,upper_right], dtype=np.int32)],
                    1, np.array([255, 0, 0]))
        cv.polylines(D.threshed_image, [np.array([upper_left,lower_left,lower_right,upper_right], dtype=np.int32)],
                    1, np.array([255, 0, 0]))

        # Draw the circle, at the image center for now (you'll change this)

        #print biggest
        #print second_biggest
        #calculate moments for biggest and second biggest blobs
        biggest_moment = cv.moments(biggest)
        second_moment = cv.moments(second_biggest)


        #if these blobs have areas > 0, then calculate the average of their centroids
        if (biggest_moment['m00'] > 0 and second_moment['m00'] > 0):
            center_x = biggest_moment['m10']/biggest_moment['m00']
            center_y = biggest_moment['m01']/biggest_moment['m00']

            second_center_x = second_moment['m10']/second_moment['m00']
            second_center_y = second_moment['m01']/second_moment['m00']

            led_check = are_these_leds(center_x, center_y, second_center_x, second_center_y)

            if (led_check):
                D.tadro_center = (int((center_x + second_center_x)/2), int((center_y + second_center_y)/2))
                cv.circle(D.image, D.tadro_center, 10, np.array([255, 255, 0]))
                cv.circle(D.threshed_image, D.tadro_center, 10, np.array([255, 255, 0]))
            else:
                D.tadro_center = None
                center = (br[0] + br[2]/2, br[1] + br[3]/2)
                cv.circle(D.image, center, 10, np.array([255, 255, 0]))
                cv.circle(D.threshed_image, center, 10, np.array([255, 255, 0]))
                
            
        #else simply calculate the centroid of the largest blob
        else:
            D.tadro_center = None
            center = (br[0] + br[2]/2, br[1] + br[3]/2)
            cv.circle(D.image, center, 10, np.array([255, 255, 0]))
            cv.circle(D.threshed_image, center, 10, np.array([255, 255, 0]))
            

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
def change_slider(name, new_threshold):
    """ a small function to change a slider value """
    # get D so that we can change values in it
    global D
    D.thresholds[name] = new_threshold


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

    find_biggest_region()
    #find_tadro()

    ############3
    '''
    WRITE THE ABOVE FUNCTION
    '''
    ################3

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
        cv.imshow('threshold', D.threshed_image)#D.current_threshold )



def handle_kinect_data(data):
    """ this function grabs images from the Kinect
    """
    # get D so that we can change values in it
    global D

    # Get the incoming image from the Kinect
    D.orig_image = D.bridge.imgmsg_to_cv(data, "bgr8")

    # now, handle that image...
    handle_image()
    

'''
def handle_camera_data(data):
    """ this function grabs images from a webcamera
    """
    # get D so that we can change values in it
    global D

    while rospy.is_shutdown() == False:

        # Get the incoming image from the missle launcher or other camera
        D.orig_image = cv.QueryFrame(D.camera)

        # now, handle that image...
        handle_image()
'''

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
        if (D.tadro_center != None):
            D.tadro_data.append((j, D.tadro_center))

        #quit if told to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        #skip FRAME_RATE number of frames
        for i in range(0, D.FRAME_RATE):
            cap.grab()

        #increment the frame counter
        j += 1 + D.FRAME_RATE

    make_tadro_path_image() #make an image displaying Tadro path

    #save the image in the current directory
    cv.imwrite("./path_image.png", D.tadro_image)

    #clean up
    cap.release()
    cv.destroyAllWindows()

# this is the "main" trick - it tells Python
# what to run as a stand-alone script:
if __name__ == "__main__":
    main()
