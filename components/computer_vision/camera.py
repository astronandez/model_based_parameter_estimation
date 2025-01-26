import json
import cv2 as cv
import time

class Camera:
    capture: cv.VideoCapture
    video_output: cv.VideoWriter
    
    def __init__(self, config: json):
        capture = cv.VideoCapture(config['video_input'])
        fw, fh, fps = (int(capture.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH,
                                                     cv.CAP_PROP_FRAME_HEIGHT, 
                                                     cv.CAP_PROP_FPS))
        video_output = cv.VideoWriter(f"./videos/recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4", 
                                      cv.VideoWriter_fourcc(*'mp4v'),
                                      fps, (config['screen_width'], config['screen_height']))
        
        self.capture = capture
        self.fw = fw
        self.fh = fh
        self.fps = fps
        self.video_output = video_output
        self.config = config
        
    def record(self):
        print("Start recording...")
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                frame = eval(self.config['rotation'])
                self.video_output.write(frame)
                cv.imshow('Recording', frame)
                # Clear video elements
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.capture.release()
                    self.video_output.release()
                    cv.destroyAllWindows()
                    break
            else:
                break
        
############ Testbench ############    
if __name__ == "__main__":
    
    def loadConfig(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    camera_config_path = './configuration/cameras/rot90clockwise.json'
    camera_config = loadConfig(camera_config_path)
    camera = Camera(camera_config)
    camera.record()