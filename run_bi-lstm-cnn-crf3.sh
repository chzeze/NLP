train="data/POS-penn/wsj/split1/wsj1.train.original"
train=$1
dev="data/POS-penn/wsj/split1/wsj1.dev.original"
dev=$2
test="data/POS-penn/wsj/split1/wsj1.test.original"
test=$3
edict="data/glove/glove.6B/glove.6B.100d.gz"
edict=$4
model=$5

update=momentum
bsize=10
#bsize=5

num_units=300
num_units=200
num_units=400
num_units=$6

THEANO_FLAGS='floatX=float32,mode=FAST_RUN,device=gpu' /usr/bin/python bi_lstm_cnn_crf3.py --fine_tune --embedding glove --oov embedding --update adadelta \
 --batch_size ${bsize} --num_units ${num_units} --num_filters 30 --learning_rate 0.015 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
 --train $train --dev $dev --test $test \
 --embedding_dict $edict --patience 5 --model $model
