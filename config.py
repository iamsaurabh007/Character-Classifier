import string
symbols=list(string.printable[:94])
symbols.append(u"\u00A9")
symbols.append(u"\u2122")
symbols.append(" ")
num_classes=len(symbols)

dir_path="/home/saurabhyadav007/Proj/data/ocr/out"
device='cpu'

#USED IN DATALOADER
batch_size=32
shuffle=True
num_workers=2


#USED IN MODEL
learning_rate=0.01
num_epochs=50


