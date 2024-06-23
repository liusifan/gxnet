which python || alias python=python3

if [ $# != 1 ];
then
	echo "Usage: $1 <jpeg image file/dir>"
	exit
fi

path=$1

match=0
total=0

files=""

if [ -d $path ];
then
	files=`ls $path/*.jpeg $path/*.jpg  2>/dev/null`
fi

if [ -f $path ];
then
	files=$path
fi

for i in $files;
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

echo "result $match / $total "

