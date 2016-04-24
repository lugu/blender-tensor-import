
all:
	mkdir -p dataset
	python import.py data/*.png

clean:
	rm -fr dataset/*

test:
	python test.py dataset/train.tfrecords
