#pip install torchinfo

#from torchinfo import summary
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
#from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from torch.autograd import Variable
import torch.onnx


parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-o", "--output", help="outputfile name", default="arcface.onnx")
parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)

args = parser.parse_args()

conf = get_config(False)

#mtcnn = MTCNN()
#print('arcface loaded')

learner = face_learner(conf, True)
learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

model = learner.model
batch_size=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use this an input trace to serialize the model
input_shape = (3, 112, 112)
dummy_input = Variable(torch.randn(1, *input_shape))
dummy_input = dummy_input.to(device)
# Export the model
torch.onnx.export(model,     # model being run
  dummy_input,                         # model input (or a tuple for multiple inputs)
  args.output,
  verbose=False)
      # where to save the model (can be a file or file-like object)
#  export_params=True,        # store the trained parameter weights inside the model file
#  opset_version=10,          # the ONNX version to export the model to
#  do_constant_folding=True,  # whether to execute constant folding for optimization
#  input_names = ['input'],   # the model's input names
#  output_names = ['output'], # the model's output names
#  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#        'output' : {0 : 'batch_size'}})

