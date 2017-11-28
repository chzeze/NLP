#! /bin/sh

#data=data_CAM/joint_allfull_new/ComponentsTypesRelations/
#data=data_CAM/paragraphs/
data=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/conll/Paragraph_Level/
nunits=200

ouputPath=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/PARA
embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.100d.txt.gz
#1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

for i in 0 1 2 3 4 5 6 7 8 9 10; do 
./run_bi-lstm-cnn-crf3.sh ${data}/train_comp_MCP.txt ${data}/dev_comp_MCP.txt ${data}/test_comp_MCP.txt ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph_comp_mcp${i}.mod 200 > ${ouputPath}/paragraph_comp_mcp_100d200_${i}.dat
done

# embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.50d.txt.gz
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 125 > ${ouputPath}/paragraph50d125.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 150 > ${ouputPath}/paragraph50d150.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 200 > ${ouputPath}/paragraph50d200.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 250 > ${ouputPath}/paragraph50d250.dat


# embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.100d.txt.gz
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 125 > ${ouputPath}/paragraph100d125.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 150 > ${ouputPath}/paragraph100d150.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 200 > ${ouputPath}/paragraph100d200.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 250 > ${ouputPath}/paragraph100d250.dat


# embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.200d.txt.gz
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 125 > ${ouputPath}/paragraph200d125.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 150 > ${ouputPath}/paragraph200d150.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 200 > ${ouputPath}/paragraph200d200.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 250 > ${ouputPath}/paragraph200d250.dat

# embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.300d.txt.gz
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 125 > ${ouputPath}/paragraph300d125.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 150 > ${ouputPath}/paragraph300d150.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 200 > ${ouputPath}/paragraph300d200.dat
# ./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/MODELS/sub_paragraph.mod 250 > ${ouputPath}/paragraph300d250.dat


#./run_bi-lstm-cnn-crf3.sh ${data}/train.dat ${data}/dev.dat ${data}/test.dat ${embeddings} MODELS/sub_paragraph.mod 150 > /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/PARA/paragraph.dat

# for i in 0 1 2 3 4; do 
  # embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.100d.txt.gz 
  # ./run_bi-lstm-cnn-crf3.sh ${data}/sub/train+dev_full.dat${i}.train ${data}/sub/train+dev_full.dat${i}.dev ${data}/test_full.dat ${embeddings} MODELS/sub_paragraph${i}.mod 150 > /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/PARA/paragraph${i}.dat
# done &

# for i in 5 6 7 8 9; do
  # embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.50d.txt.gz
  # ./run_bi-lstm-cnn-crf3.sh ${data}/sub/train+dev_full.dat${i}.train ${data}/sub/train+dev_full.dat${i}.dev ${data}/test_full.dat ${embeddings} MODELS/sub_paragraph${i}.mod 200 > /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/PARA/paragraph${i}.dat
# done &


# for i in 10 11 12 13 14; do
  # embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.200d.txt.gz
  # ./run_bi-lstm-cnn-crf3.sh ${data}/sub/train+dev_full.dat${i}.train ${data}/sub/train+dev_full.dat${i}.dev ${data}/test_full.dat ${embeddings} MODELS/sub_paragraph${i}.mod 250 > /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/PARA/paragraph${i}.dat
# done &

# for i in 15 16 17 18 19; do
  # embeddings=/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.50d.txt.gz
  # ./run_bi-lstm-cnn-crf3.sh ${data}/sub/train+dev_full.dat${i}.train ${data}/sub/train+dev_full.dat${i}.dev ${data}/test_full.dat ${embeddings} MODELS/sub_paragraph${i}.mod 125 > /home/fzuir/czz/acl2017-neural_end2end_am-master/data/outputs_ACL/PARA/paragraph${i}.dat
# done &

