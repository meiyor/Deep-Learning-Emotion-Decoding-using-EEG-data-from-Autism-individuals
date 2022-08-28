## generate code for LOTO processing using python and the CNN code template from FAIR using Tensorflow##
import os
import sys
import subprocess

# this for loop execute the Leave-One-Trial-Out cross-validation please set the clinical participant files following the README instructions
for root, dirnames, filenames in os.walk(str(sys.argv[1])):
    for name in sorted(dirnames):
        print(name)
        file_name = sorted(os.listdir(''.join([str(sys.argv[1]), '/', name])))
        print(file_name)
        for files in file_name:
            print(''.join([str(sys.argv[1]), '/', name, '/', files]))
            if os.path.isfile(''.join([str(sys.argv[1]), '/', name, '/', files])):
                # before run each subjects run counter_divide_loto.sh and creates all the temporary training and test files, ALL THE TEMPORARY FILES SHOULD BE CREATED A PRIORI
                # define the result folder per subject ONLY USE THIS TO TEST THE NEXT SUBJECT AND DEFINE THE RECTANGULAR KERNEL SIZE TEST.
                if "train" in files and os.path.isdir(''.join([str(sys.argv[1]), '/', name, '/##folder_name##'])) == 0:
                    str1 = ''.join([str(sys.argv[1]), '/', name, '/', files])
                    # define this number depending on the number of /s you have in your data path
                    str_test = "test"+files[5:]
                    print(str_test)
                    str2 = ''.join(
                        [str(sys.argv[1]), '/', name, '/', str_test])
                    print(str1, str2)
                    # run the process the emotion decoding as a whole the stdout performance
                    subprocess.call('python CNN_launching_classifier_emotions.py "%s" "%s" 0 "%s"' % (str1, str2, str(sys.argv[2])), shell=True)
