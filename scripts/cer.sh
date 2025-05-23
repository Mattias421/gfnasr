sclite -r $1 -h $2 -i spu_id -c -o sum stdout | grep 'Sum/Avg' | awk '{print $10}'

