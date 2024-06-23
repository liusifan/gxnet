which python || alias python=python3

python rotate_mnist.py

./testmnist -l 10

sh testuat.sh uat
sh testuat.sh uat/ian

./testmnist -l 2 -p mnist.model

sh testuat.sh uat
sh testuat.sh uat/ian

