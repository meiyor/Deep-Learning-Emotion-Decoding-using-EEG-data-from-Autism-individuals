line_p=$(awk '/trial8/ { ln = FNR } END { print ln }' $1)
echo $line_p
sed -i "${line_p},\$d" $1
line_q=$(awk '/trial9/ { ln = FNR } END { print ln }' $1)
echo $line_q
line_q=$((line_q + 3))
sed -i "${line_q},\$d" $1
