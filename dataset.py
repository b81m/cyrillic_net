from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
from PIL import Image

def invert_colors(x):
    return 1.0 - x

def white_bg_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            
            img = img.convert('RGBA')
            
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            
            background.paste(img, (0, 0), img)
            
            return background.convert('RGB')
        
def get_dataloaders(image_size=28,num_channels=1,path_to_data="Cyrillic"):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.Resize((image_size,image_size)),
        transforms.RandomAffine(
            degrees=12,
            translate=(0.1,0.1), #offset from center by x,y coordinates
            scale=(0.9,1.1), #zoom image from 85% to 115%
            shear=10,
            fill = 255
        ),
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=50.0,sigma=5.0,fill=50)
        ],p=0.1),
        transforms.ToTensor(),
        transforms.Lambda(invert_colors),

        transforms.RandomErasing(p=0.5,scale=(0.02,0.1),value=0),

        # transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset=datasets.ImageFolder(root=path_to_data,
                             transform=transform,
                             loader=white_bg_loader)

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset,test_dataset = random_split(dataset,[train_size,test_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=256,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                            num_workers=2,
                              pin_memory=True)  

    return train_loader,test_loader,dataset.classes

train_loader,test_loader,dataset = get_dataloaders()

