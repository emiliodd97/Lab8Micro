// Christian Medina Armas
// Emilio Diaz
//15316
// CC3056
// CUDA
// Apply a filter to an image

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include "kernel.h"

using namespace cv;
using namespace std;



int main(){
//Lectura de la imagen 
	Mat input_img;
	input_img = imread("ramphastosSulphuratus.jpeg", CV_LOAD_IMAGE_GRAYSCALE);

	// create a zero filled Mat of the input image size
	//Matriz de salida
	Mat output_img = Mat::zeros(Size(input_img.cols, input_img.rows), CV_8UC1);

	// compute filter
	//Inicio de toma de tiempo
	double time = (double) getTickCount;
	//aplica filtro
	filter_gpu(input_img, output_img);
	//imprime el tiempo
	time = ((double) getTickCount() - time)/getTickFrequency();
	cout << "Tiempo total: "<< time<< "segundos"<<endl;
	

//Guardar imagen de salida
	imwrite("Filter_EmilioDiaz.jpg", output_img);
	return 0;

}
