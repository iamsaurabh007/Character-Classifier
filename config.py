import string
symbols=list(string.printable[:94])
symbols.append(u"\u00A9")
symbols.append(u"\u2122")
symbols.append(" ")
num_classes=len(symbols)

dir_path="/home/ubuntu/data/ocr/out"
csv_path='/home/ubuntu/Character-Classifier/hypergridcsv'
device=None

#USED IN DATALOADER
batch_size=16
shuffle=True
num_workers=6


#USED IN MODEL
learning_rate=0.001
num_epochs=50

#channels
channel=32


