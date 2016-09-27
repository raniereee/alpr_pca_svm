all:
	g++ codigo.cpp -o alprpca `pkg-config --libs --cflags opencv`
