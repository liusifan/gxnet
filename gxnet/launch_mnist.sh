which python > /dev/null 2>&1|| alias python=python3

myunzip()
{
	if [ ! -f $1 ];
	then
		gunzip -k $1.gz
	fi
}

train_images="mnist/train-images-idx3-ubyte"
train_labels="mnist/train-labels-idx1-ubyte"

test_images="mnist/t10k-images-idx3-ubyte"
test_labels="mnist/t10k-labels-idx1-ubyte"

myunzip $train_images
myunzip $train_labels
myunzip $test_images
myunzip $test_labels

rot_images=$train_images".rot"
rot_labels=$train_labels".rot"

if [ ! -f $rot_images ];
then
	python trans_mnist.py rotate $train_images $train_labels
fi

./testmnist --lr 5

sh testuat.sh -m mnist.model -p uat/digits
sh testuat.sh -m mnist.model -p uat/digits/ian

./testmnist --lr 2 --model mnist.model

sh testuat.sh -m mnist.model -p uat/digits
sh testuat.sh -m mnist.model -p uat/digits/ian

