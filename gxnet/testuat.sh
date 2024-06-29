which python > /dev/null 2>&1|| alias python=python3

if [ $# -eq 0 ];
then
	echo "Usage: $1 <jpeg image file/dir> [<model file>] [<uat suffix>]"
	exit
fi

path=$1
model="mnist.model"
suffix="mnist"

if [ $# -gt 1 ];
then
	model=$2
fi

if [ $# -gt 2 ];
then
	suffix=$3
fi

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
	target=`echo $i | grep -Eo '([0-9]+)' | head -1`

	test -f $i"."$suffix || python ./conv2mnist.py $i

	test -f $i"."$suffix && ./gxocr $model $i"."$suffix

	if [ "$target" -eq "$?" ];
	then
		match=$((match+1))
	fi

	total=$((total+1))
done

echo "result $match / $total "

