import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models




def arg_parser():
    
    parser = argparse.ArgumentParser(description = "for predicting the image")
    
    parser.add_argument("--image",
                        type = str,
                        help = "path to image",
                        required = True)
    parser.add_argument("--top_k",
                        type = int,
                        help = "top prob")
    parser.add_argument("--checkpoint",
                        type = str,
                        help = "path to checkpoint",
                        required = True)
    parser.add_argument("--category_names",
                        type = str,
                        help = "for mapping names")
    parser.add_argument("--image",
                        type = str,
                        help = "path to image")
    parser.add_argument('--gpu',
                        action = 'store_true',
                        help = "use gpu for training the network")
    args = parser.parse_args()
    return args


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    if img.size[0]>img.size[1]:
        img.thumbnail((10000,256))
    else:
        img.thumbnail((256,20000))
    # TODO: Process a PIL image for use in a PyTorch model
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin+224
    top_margin = bottom_margin+224
    
    img = img.crop((left_margin,bottom_margin,right_margin,top_margin))
    
    img = np.array(img)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img-mean)/std
    
    img = img.transpose((2,0,1))
    return img

def load_model(checkpoint_pth):
    cht = torch.load(checkpoint_pth)
    
    model = getattr(models,cht['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False    
        
    model.classifier = cht["classifier"]
    model.class_to_idx = cht['class_to_idx']
    model.load_state_dict(cht['state_dict'])
    return model

def predict(image_path, model, topk=5,cat_to_name,device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    with torch.no_grad():
        img = process_image(image_path)
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        inp = img_tensor.unsqueeze(0)
        model.to(device),inp.to(device)
        logps = model(inp)
        ps = torch.exp(logps)
        top_ps,top_class = ps.topk(topk)
        top_ps = top_ps.numpy().tolist()[0] 
        top_class = top_class.numpy().tolist()[0]
        
        idx_to_class = {val:key for key,val in model.class_to_idx.items()}
        
        top_labels = [idx_to_class[lab] for lab in top_class]
        top_flower = [cat_to_name[idx_to_class[lab]] for lab in top_class]
        
        return top_ps,top_labels,top_flower

def print_probability(probs, flowers):
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
    
def main():
    args = args_parser()
    
    with open(args.category_names,'r') as f:
        cat_to_name = json.load(f)
        
    model = load_model(args.checkpoint)
    device = check_gpu(gpu_arg=args.gpu);
    image_path = args.image
    top_k = args.top_k
    top_ps,top_labels,top_flowers = predict(image_path,model,top_k,cat_to_name,device)
    
    print_probability(top_flowers, top_ps)
    
if __name__ == '__main__':
    main()
    