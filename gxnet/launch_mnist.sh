which python > /dev/null 2>&1|| alias python=python3

train_images="mnist/train-images.idx3-ubyte"
train_labels="mnist/train-labels.idx1-ubyte"

test_images="mnist/t10k-images.idx3-ubyte"
test_labels="mnist/t10k-labels.idx1-ubyte"

rot_images=$train_images".rot"
rot_labels=$train_labels".rot"

if [ ! -f $rot_images ];
then
	python trans_mnist.py rotate $train_images $train_labels
fi

./testmnist -l 5

sh testuat.sh uat mnist.model mnist
sh testuat.sh uat/ian mnist.model mnist

./testmnist -l 2 -p mnist.model

sh testuat.sh uat mnist.model mnist
sh testuat.sh uat/ian mnist.model mnist

