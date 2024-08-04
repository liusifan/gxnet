which python > /dev/null 2>&1|| alias python=python3

PROG=$0

path=""
model="mnist.model"

OPTSTRING=":m:p:"

while getopts ${OPTSTRING} opt; do
	case ${opt} in
		m)
			model=${OPTARG}
			;;
		p)
			path=${OPTARG}
			;;
	esac
done

if [ "" = "$path" ];
then
	echo "Usage: $PROG -m <model path> -p <image file/dir>"
	exit
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

	test -f $i".mnist" || python ./conv2mnist.py $i

	test -f $i".mnist" && ./gxocr --model $model --file $i".mnist"

	if [ "$target" -eq "$?" ];
	then
		match=$((match+1))
		printf "\033[1A\033[70C\033[;32mPassed\033[0m\n"
	else
		printf "\033[1A\033[70C\033[;31mFailed\033[0m\n"
	fi

	total=$((total+1))
done

echo "result $match / $total "

