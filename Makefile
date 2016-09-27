all:
	g++ trainPCA4SVM.cpp -o trainPCA4SVM `pkg-config --libs --cflags opencv`
