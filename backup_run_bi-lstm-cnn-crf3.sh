#train="data/POS-penn/wsj/split1/wsj1.train.original"
train="/home/fzuir/czz/acl2017-neural_end2end_am-master/data/conll/Essay_Level/train.dat"
#dev="data/POS-penn/wsj/split1/wsj1.dev.original"
dev="/home/fzuir/czz/acl2017-neural_end2end_am-master/data/conll/Essay_Level/dev.dat"
#test="data/POS-penn/wsj/split1/wsj1.test.original"
test="/home/fzuir/czz/acl2017-neural_end2end_am-master/data/conll/Essay_Level/test.dat"
edict="/home/fzuir/czz/acl2017-neural_end2end_am-master/data/glove.6B/glove.6B.100d.txt.gz"
#edict=$4
#model=$5

update=momentum
bsize=10
#bsize=5

num_units=300
num_units=200
num_units=400
#num_units=$6

model="Test"

THEANO_FLAGS='floatX=float32,mode=FAST_RUN,device=gpu' /usr/bin/python bi_lstm_cnn_crf3.py --fine_tune --embedding glove --oov embedding --update adadelta \
 --batch_size ${bsize} --num_units ${num_units} --num_filters 50 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
 --train $train --dev $dev --test $test \
 --embedding_dict $edict --patience 5 --model $model
