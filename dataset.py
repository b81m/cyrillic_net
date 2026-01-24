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
        
def get_dataloaders(path_to_data="Cyrillic"):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1,0.1), #offset from center by x,y coordinates
            scale=(0.85,1.15), #zoom image from 85% to 115%
                fill = 255
        ),
        transforms.ToTensor(),
        transforms.Lambda(invert_colors),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset=datasets.ImageFolder(root=path_to_data,
                             transform=transform,
                             loader=white_bg_loader)

    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset,test_dataset = random_split(dataset,[train_size,test_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=64,
                             shuffle=False)

    return train_loader,test_loader,dataset.classes
