which python || alias python=python3

if [ $# != 1 ];
then
	echo "Usage: $1 <jpeg image dir>"
	exit
fi

dir=$1

match=0
total=0

for i in `ls $dir/*.jpeg $dir/*.jpg`;
do
	target=`echo $i | grep -o '[0-9]' | head -1`

	test -f $i".mnist" || python ./conv2mnist.py $i

	test -f $i".mnist" && ./gxocr mnist.model $i".mnist"

	if [ "$target" -eq "$?" ];
	then
		match=$((match+1))
	fi

	total=$((total+1))
done

echo "$match / $total "

