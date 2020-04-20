import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torchvision import datasets, transforms, models
from torch import nn,optim
from workspace_utils import active_session


def args_parse():
    parser = argparse.ArgumentParser(description = 'for training neural network')
    
    parser.add_argument('--arch',
                        type = str,
                        help = 'choosing the architecture' )
    parser.add_argument('--hidden_units',
                        type = int,
                        help = 'vhoosing the hidden unit for the architecture')
    parser.add_argument('--epochs',
                        type = int,
                        help = "no. of epochs for training the model")
    parser.add_argument('--learning_rate',
                        type = float,
                        help = 'learning rate for the network')
    parser.add_argument('--gpu',
                        action = 'store_true',
                        help = "use gpu for training the network")
    parser.add_argument('--save_dir',
                        type = str,
                        help = 'directory to save the model checkpoint')
    args = parser.parse_args()
    return args

def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
    return train_data

def test_transformer(valid_dir,test_dir):
    valid_and_test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    valid_data = datasets.ImageFolder(valid_dir,transform = valid_and_test_transform)
    test_data = datasets.ImageFolder(test_dir,transform = valid_and_test_transform)
    return valid_data,test_data

def data_load(data,train = True):
    if train:
        loader = torch.utils.data.DataLoader(data,batch_size = 32,shuffle = True)
    else:
        loader = torch.utils.data.DataLoader(data,batch_size = 32)
    return loader

def arch_of_classifier(architecture):
    if type(architecture)==type(None):
        model = models.vgg19(pretrained==True)
        model.name = "vgg19"
    else:
        model = getattr(models,architecture)(pretrained = True)
        model.name = architecture
    for param in model.parameters():
        param.requires_grad = False
    return model

def model_classifier(model,hidden_units):
    if type(hidden_units) == type(hidden_units):
        hidden_units = 4096
    input = model.classifier[0].in_features
    model.classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(input,hidden_units)),
                                                      ('relu',nn.ReLU()),
                                                      ('Dropout',nn.Dropout(p=0.2)),
                                                      ('fc2',nn.Linear(hidden_units,102)),
                                                      ('softmax',nn.LogSoftmax(dim=1))]))
    return model

def train_model(trainloader,validloader,device,epochs,learning_rate,model,optimizer,criterion):
    if type(epochs) == type(None):
        epochs = 10
    steps_every=5
    for epoch in range(epochs):
        running_loss = 0
        steps = 0
        for images,labels in trainloader:
            steps+=1
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
                
            if steps%steps_every==0:
               accuracy = 0
               valid_loss = 0
               model.eval()
               with torch.no_grad():
                    for images,labels in validloader:
                        images,labels = images.to(device),labels.to(device)
                        logps = model(images)
                        loss = criterion(logps,labels)
                        valid_loss+=loss.item()
                        ps = torch.exp(logps)
                        top_ps,top_labels = ps.topk(1,dim=1)
                        equals = top_labels == labels.view(*top_labels.shape)
                        accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                     
               model.train()
               print("epochs :{}/{}".format(epoch+1,epochs),
                     "Running_loss:{}|".format(running_loss/steps_every),
                     "valid_loss:{}|".format(valid_loss/len(validloader)),
                     "accuracy:{}|".format(accuracy/len(validloader)*100))     
       
    return model             
def check_gpu(gpu):
    if not gpu:
        print("cpu is using")
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    return device

def save_checkpoint(save_dir,model,train_data):
    if type(save_dir)==type(None):
        print("directory is not defined,model will not be saved")
    else:
        if isdir(save_dir):
            model.class_to_idx = train_data.class_to_idx
            model.cpu()
            torch.save({'arch':model.name,
                        "classifier":model.classifier,
                        "state_dict":model.state_dict(),
                        "class_to_idx":model.class_to_idx},
                        "classifier.pth")
        else:
            print("directory is not availiable")
def valid(testloader,device,model):
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images,labels in testloader:
            images,labels = images.to(device),labels.to(device)
            logps = model(images)
            ps = torch.exp(logps)
            top_ps,top_class = ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
    model.train()
    print("accuracy: {}".format(accuracy/len(testloader)*100))
  
         

def main():
    
    args = args_parse()
    
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transformer(train_dir)
    valid_data,test_data = test_transformer(valid_dir,test_dir)
    
    trainloader = data_load(train_data)
    validloader = data_load(train_data,train = False)
    testloader = data_load(train_data,train = False)
    
    model_arch =arch_of_classifier(architecture=args.arch)
    model = model_classifier(model_arch,args.hidden_units)
                             
    device = check_gpu(gpu=args.gpu);
    
    model.to(device);
    
    if type(args.learning_rate) == type(None):
        learning_rate == 0.002
    else:
        learning_rate = args.learning_rate
        
    optimizer = optim.Adam(model.classifier.parameters(),learning_rate)
    criterion = nn.NLLLoss()
    epochs = args.epochs
    with active_session():
        trained_model = train_model(trainloader,validloader,device,epochs,learning_rate,model,optimizer,criterion)
    
    print("training is done")
    with active_session():
        
        valid(testloader,device,trained_model)
    
    print("inference is complete")
    
    save_checkpoint(args.save_dir,trained_model,train_data)
    
    print("model is saved")
    
    
if __name__ == '__main__':
    main()