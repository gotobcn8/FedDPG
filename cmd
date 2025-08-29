nohup python main.py --dataset sst2 --unlearn_method fast --device cuda:0 > sst2-fast.log 2>&1 &
nohup python main.py --dataset agnews --unlearn_method fast --device cuda:1> agnews-fast.log 2>&1 &
nohup python main.py --dataset yelp --unlearn_method fast --device cuda:2 > yelp-fast.log 2>&1 &

nohup python main.py --dataset sst2 --unlearn_method relabel --device cuda:3 > sst2-relabel.log 2>&1 &
nohup python main.py --dataset agnews --unlearn_method relabel --device cuda:0 > agnews-relabel.log 2>&1 &
nohup python main.py --dataset yelp --unlearn_method relabel --device cuda:1 > yelp-relabel.log 2>&1 &