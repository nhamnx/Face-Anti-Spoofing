# from facenet_pytorch import MTCNN
import time
import cv2 as cv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from model import DeePixBiS
from loss import PixWiseBCELoss
from metrics import predict, test_accuracy, test_loss
from facenet_pytorch import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS_compose_oldaug.pth'))
model.eval()
model.to(device)


tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

res_dict = {}

mtcnn = MTCNN(device = 'cuda:0')
def cam_capture(cam_url):
    count = 3
    i = 0
    
    camera = cv.VideoCapture(cam_url)
    prev_frame_time = 0
    new_frame_time = 0
    while cv.waitKey(1) & 0xFF != ord('q'):
        success , frame = camera.read()
        if not success:
            pass
        else:
            
            rate = 3
            small_frame = cv.resize(frame, (0, 0), fx=round(1/rate, 2), fy=round(1/rate, 2))
            norm_img = np.zeros((small_frame.shape[0], small_frame.shape[1]))
            norm_small_frame = cv.normalize(small_frame, norm_img, 0, 255, cv.NORM_MINMAX)
            small_rgb_frame = cv.cvtColor(norm_small_frame, cv.COLOR_BGR2RGB)
            # try:
            faces, probs = mtcnn.detect(small_rgb_frame)
            color = (255,0,0)
            if faces is not None:
                faces = faces.astype('int32')
                for(face, prob) in zip(faces, probs):
                    if prob > 0.9:
                        x1, y1, x2, y2 = face
                        X1, Y1, X2, Y2 = x1*rate, y1*rate, x2*rate, y2*rate 
                        try:

                            if count % 3 == 0:
                                count = 0
                                faceRegion = frame[Y1:Y2, X1:X2]
                                faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
                                faceRegion = tfms(faceRegion)
                                faceRegion = faceRegion.unsqueeze(0)
                                mask, binary = model.forward(faceRegion.to(device))
                                res = torch.mean(mask).item()
                                res_dict[f"face{i}"] = res

                            if res_dict[f"face{i}"] < 0.8:
                                color = (0, 0, 255)
                                cv.putText(frame, 'Fake', (X1, Y2 + 30), cv.FONT_HERSHEY_COMPLEX, 1, color)
                            else:
                                color = (0, 255, 0)
                                cv.putText(frame, 'Real', (X1, Y2 + 30), cv.FONT_HERSHEY_COMPLEX, 1, color)
                    
                        except:
                            pass
                        cv.rectangle(frame, (X1, Y1), (X2 , Y2 ), color, 2)
                    i+=1
                i = 0
            count += 1
            
            
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
        
            fps = int(fps)
        
            fps = str(fps)
        
            cv.putText(frame, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 2, cv.LINE_AA)
            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             cv.imshow('test', frame)

# cam_capture(0)






