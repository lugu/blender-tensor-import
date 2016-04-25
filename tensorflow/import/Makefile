
all:
	mkdir -p dataset
	python import.py data/*.png

clean:
	rm -fr dataset sum

test:
	rm -fr sum
	python test.py dataset/train.tfrecords
	tensorboard --logdir=sum --host 127.0.0.1 >/dev/null 2>&1 &
	surf "http://127.0.0.1:6006" > /dev/null 2>&1
