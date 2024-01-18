#! /usr/bin/bash 

export COMBINE=/home/ubuntu/combine.py
# combine less32_test and less1_validation => new test split 
python3 ${COMBINE} --file1 /home/ubuntu/less27_test.txt --file2 /home/ubuntu/less27_validation.txt --destination /home/ubuntu/new_test.txt  

# combine over32_test and over1_validation => new validation split 
python3 ${COMBINE} --file1 /home/ubuntu/over27_test.txt --file2 /home/ubuntu/over27_validation.txt --destination /home/ubuntu/new_validation.txt 