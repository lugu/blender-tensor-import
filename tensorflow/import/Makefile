
all: dataset
	mkdir -p dataset

dataset/train.tfrecords:
	mkdir dataset
	python import.py --directory dataset data/*.png

clean:
	rm -fr dataset test summary

summary: test.py dataset/train.tfrecords
	python test.py --summary summary --dataset dataset/train.tfrecords

view: summary
	tensorboard --logdir=summary --host 0.0.0.0 >/dev/null 2>&1
	# surf "http://127.0.0.1:6006" > /dev/null 2>&1

test/data:
	rm -fr test; mkdir -p test/data
	cd test/data; for i in $$(seq 0 9); do \
	convert -geometry 128 -background lightblue -fill red  -pointsize 72 label:$$i image-$$i.Left.png; \
	convert -geometry 128 -background lightblue -fill green  -pointsize 72 label:$$i image-$$i.Right.png; \
	convert -geometry 128 -background lightblue -fill blue  -pointsize 72 label:$$i image-$$i.Left-depth.png; \
	convert -geometry 128 -background lightblue -fill yellow  -pointsize 72 label:$$i image-$$i.Right-depth.png; \
	done

test/dataset/train.tfrecords: test/data import.py
	mkdir -p test/dataset
	python import.py --directory test/dataset test/data/*.png

test/summary: test.py test/dataset/train.tfrecords
	python test.py --summary test/summary --dataset test/dataset/train.tfrecords
	tensorboard --logdir=test/summary --host 0.0.0.0 >/dev/null 2>&1
	# surf "http://127.0.0.1:6006" > /dev/null 2>&1

testsuite: test/summary
