import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import sys
from face_recognition import face_compare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    img = []
    
    img.append(cv2.imread('data/input/evans/evans_p.jpg'))
    img.append(cv2.imread('data/input/hermsworth/hermsworth_p.jpg'))
    img.append(cv2.imread('data/input/jeremy/jeremy.jpg'))
    img.append(cv2.imread('data/input/mark/mark.jpg'))
    img.append(cv2.imread('data/input/olsen/olsen.jpg'))

    
    for i in range(5):
        try:
            image = Image.fromarray(img[i][...,::-1]) #bgr to rgb
#            image = Image.fromarray(img[i])
            print('----------------------------------')
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice   
            results, score = face_compare(conf, learner.model, faces, targets, args.tta)
            num_face = len(results) #len(results)가 얼굴개수가나오므로 num_face라는 변수 서연이 만듬. 
            print(num_face)
            for idx,bbox in enumerate(bboxes):
                if args.score: 
                    #args.score는 주로 false로 나오기때문에 boundingbox옆에 score가 나오게 하려면 else쪽으로 넣어야함.
                    img[i] = draw_box_name(bbox, names[results[idx] + 1], img[i])
                else:
                    img[i] = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), img[i])
        except:
            print('detect error')    
            
        cv2.imwrite('data/output/img_{}.jpg'.format(i), img[i])
        
