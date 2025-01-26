import cv2 as cv
from numpy import ndarray

def centerToBoundingBox(cx, cy, w, h):
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    
    return x1, y1, x2, y2

class Display:
    def __init__(self):
        pass
    
    def draw(self, frame: cv.Mat, H: ndarray, Q: ndarray, R:ndarray, mass: float, zx: float, zy: float, 
             width: float, height: float, color: tuple, sensor_time: float, offset: float):
        
        x1, y1, x2, y2 = centerToBoundingBox(zx, zy, width, height)
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv.putText(frame, f'z: {zy} px', (int(zx), int(zy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        curr_time = '{:02d}:{:02d}:{:02d}'.format(int(sensor_time / 3600), int((sensor_time % 3600) / 60), int(sensor_time % 60))
        cv.putText(frame, curr_time, (5, 35), cv.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)
        cv.putText(frame, f'Mass Estimation: {mass} kg', (5, 545), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        final_frame = cv.putText(frame, f'zerod @: {offset}', (5, 565), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.putText(frame, f'Q: {Q}', (5, 585), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.putText(frame, f'R: {R}', (5, 605), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.line(frame, (0, offset), (480, offset), color)
        return final_frame 