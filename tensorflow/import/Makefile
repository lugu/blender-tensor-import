
all:
	mkdir -p dataset
	python import.py data/*.png

clean:
	rm -fr dataset sum

test:
	rm -fr sum
	python test.py --dataset dataset/train.tfrecords
	tensorboard --logdir=sum --host 127.0.0.1 >/dev/null 2>&1 &
	surf "http://127.0.0.1:6006" > /dev/null 2>&1

testdata:
	mkdir testdata; cd testdata; for i in $$(seq 0 9); do \
	convert -geometry 128 -background lightblue -fill red  -pointsize 72 label:$$i image-$$i.Left.png; \
	convert -geometry 128 -background lightblue -fill green  -pointsize 72 label:$$i image-$$i.Right.png; \
	convert -geometry 128 -background lightblue -fill blue  -pointsize 72 label:$$i image-$$i.Left-depth.png; \
	convert -geometry 128 -background lightblue -fill yellow  -pointsize 72 label:$$i image-$$i.Right-depth.png; \
	done

testsuite: testdata
	mkdir -p testdata/dataset
	python import.py --directory testdata/dataset testdata/*.png
	python test.py --dataset testdata/dataset/train.tfrecords
	tensorboard --logdir=sum --host 127.0.0.1 >/dev/null 2>&1 &
	surf "http://127.0.0.1:6006" > /dev/null 2>&1
