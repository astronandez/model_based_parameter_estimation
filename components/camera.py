import json
import cv2 as cv
import time
from tools.common import loadConfig

class Camera:
    input: str
    path: str
    frame_w: int
    frame_h: int
    
    # Assign object parameters using config file
    def __init__(self, config: json):
        self.input = config['input']
        self.path = config['output_path']
        self.frame_w = config['frame_w']
        self.frame_h = config['frame_h']
        self.write = config['write']
    
    # Initilization function for recording functions, this implementation avoids opening the camera
    # automatically when the camera object is initialized
    def initRecording(self):
        """
        This function initializes the camera's capture and output objects necessary to start
        and save recording files
        
        self.input: This can be set to a numerical value to use a connected camera or file path to use a video file as input
        
        """
        print("Initializing Camera...")
        capture = cv.VideoCapture(self.input)
        self.capture = capture
        fw, fh, fps = (int(capture.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH,
                                             cv.CAP_PROP_FRAME_HEIGHT, 
                                             cv.CAP_PROP_FPS))
        
        if self.write:
            final_path = f"{self.path}recording_{time.strftime('%Y%m%d_%H%M%S')}"
            print(f"Initializing Video Writer, save location at: {final_path}.mp4")
            output = cv.VideoWriter(f"{final_path}.mp4", 
                                    cv.VideoWriter_fourcc(*'mp4v'), fps, (self.frame_w, self.frame_h))
            self.path = final_path
            self.output = output
    
    def startRecording(self):
        """
        This function starts recording using the camera object initialized through
        the initRecording function and applies frame transformations through processFrame.
        """
        print("Start recording...")
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                # Main loop, modify for additional behaviors
                self.processFrame(frame)
                # Closing and Releasing video objects and closing windows to ensure corruption free saving
                if cv.waitKey(1) & 0xFF == ord("q"): 
                    self.capture.release()
                    if self.write:
                        self.output.release()
                    cv.destroyAllWindows()
                    break
            else:
                break
    
    def processFrame(self, frame):
        """
        This function is the default behavior of a camera object rotated 90 deg
        for addtional processing functionality override function
        
        frame: frame from video feed or video input file

        """
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        
        # param write is set through config file
        if self.write:
            self.output.write(frame)
            
        cv.imshow('Recording', frame)  
        return frame
            
if __name__ == "__main__":
    config_path = './configuration_files/camera_configs/camera_base.json'
    config = loadConfig(config_path)
    camera = Camera(config)
    camera.initRecording()
    camera.startRecording()