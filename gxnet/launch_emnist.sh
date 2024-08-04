which python || alias python=python3

myunzip()
{
	if [ ! -f $1 ];
	then
		gunzip -k $1.gz
	fi
}

train_images="emnist/train-images-idx3-ubyte"
train_labels="emnist/train-labels-idx1-ubyte"

test_images="emnist/test-images-idx3-ubyte"
test_labels="emnist/test-labels-idx1-ubyte"

myunzip $train_images
myunzip $train_labels
myunzip $test_images
myunzip $test_labels

./testemnist --lr 2

sh testuat.sh -m emnist.model -p uat/letters
sh testuat.sh -m emnist.model -p uat/letters/ian emnist.model

./testemnist --lr 1 --model emnist.model

sh testuat.sh -m emnist.model -p uat/letters
sh testuat.sh -m emnist.model -p uat/letters/ian emnist.model

./testemnist --lr 0.5 --model emnist.model

sh testuat.sh -m emnist.model -p uat/letters
sh testuat.sh -m emnist.model -p uat/letters/ian emnist.model

