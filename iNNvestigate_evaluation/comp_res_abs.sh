str_res1=$(/bin/grep -n "dict" file_report.txt)
> read_temp_test_vn.txt
/bin/grep -n "probabilities" file_report.txt > read_temp_test_vn.txt
str_res2=$(/bin/tail -n1 read_temp_test_vn.txt)
nc=1
prob_Acc='Accuracy ='
loss_val='loss ='
for val1 in $str_res1
do
    echo $val1
	if [ $nc == 16 ];then
	    prob_Acc="$prob_Acc $val1"
	fi
	if [ $nc == 22 ];then
	    loss_val="$loss_val $val1"
	fi
	nc=$(( $nc + 1 ))
	#echo $nc
done
echo $prob_Acc
echo $loss_val
pc=1
prob_vals='prob=['
for val2 in $str_res2
do
    echo $val2
	if [ $pc == 10 ];then
	    prob_vals="$prob_vals ${val2:2:10}"
	fi
	if [ $pc == 11 ];then
	    prob_vals="$prob_vals $val2"
	fi
	if [ $pc == 12 ];then
	    prob_vals="$prob_vals $val2"
	fi
	if [ $pc == 13 ];then
	    prob_vals="$prob_vals ${val2:0:10} ]"
	fi
	pc=$(( $pc + 1 ))
        echo $pc
done
echo $prob_vals
echo -e "trial$2\t" >> "$1_t_performance_results_innvestigate.csv"
echo -e "$prob_Acc ${loss_val:0:14}, $prob_vals\t" >> "$1_t_performance_results_innvestigate.csv"
