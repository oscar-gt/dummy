/**M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;

    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}

std::vector<String> readClassNames(const char *filename = "synset_words.txt")
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
		system("pause");
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return classNames;
}

int main(int argc, char **argv)
{
    String modelTxt = "bvlc_googlenet.prototxt";
    String modelBin = "bvlc_googlenet.caffemodel";
    String imageFile = (argc > 1) ? argv[1] : "space_shuttle.jpg";
	std::cout << "--- Image name: " << imageFile << std::endl;

	// ------------------------------------------------------------------------
	// ------------------- Modifying to import Torch7 model -------------------
	// ------------------------------------------------------------------------
	Ptr<dnn::Importer> importer;
	bool importTorchModel = true;
	bool isDataBinary = false;

	// Ascii file modified to work with torch_importer.cpp
	bool useModifiedAscii = true;
	// Torch model data in the current directory
	string modelToUse = "";
	string torchModelBinary = "CNN3_p8_n8_split4_073000.t7";
	string torchModelAscii = "CNN3_p8_n8_split4_073000.ascii.t7";
	string torchModelAsciiModified = "CNN3_mod_p8_n8_split4_073000.ascii.t7";
	if (useModifiedAscii)
	{
		modelToUse = torchModelAsciiModified;
	}
	else
	{
		modelToUse = torchModelAscii;
	}

	// ----------------------- Importing Torch7 model -------------------------
	if (importTorchModel)
	{
		try
		{
			// Loads Torch7 model. "true" indicates binary data, 
			// "false" indicates ASCII data. 
			importer = dnn::createTorchImporter(modelToUse, isDataBinary);

		}
		catch (const cv::Exception &err)
		{
			std::cerr << err.msg << std::endl;
		}
		
	}
	// ----------------- Else importing Caffe model (original code) --------------
	else
	{
		//! [Create the importer of Caffe model]
		Ptr<dnn::Importer> importer;
		try                                     //Try to import Caffe GoogleNet model
		{
			importer = dnn::createCaffeImporter(modelTxt, modelBin);
		}
		catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
		{
			std::cerr << err.msg << std::endl;
		}
		//! [Create the importer of Caffe model]

		if (!importer)
		{
			std::cerr << "Can't load network by using the following files: " << std::endl;
			std::cerr << "prototxt:   " << modelTxt << std::endl;
			std::cerr << "caffemodel: " << modelBin << std::endl;
			std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
			std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
			system("pause");
			exit(-1);
		}

	}

	// -------------------------------------------------------------------------
	// ---------------------- Checking if torch importer worked ----------------
	// -------------------------------------------------------------------------
	if (!importer)
	{
		std::cerr << "--- Can't load Torch7 model" << std::endl;
		system("pause");
		exit(-1);
	}
	else
	{
		std::cerr << "--- Torch7 importer works!!!!!!!!!" << std::endl;
		std::cerr << "--- Torch7 model, " << modelToUse << ", isBinary == "
			<< isDataBinary << std::endl;
	}

	dnn::initModule();
    

    //! [Initialize network]
    dnn::Net net;
	std::cerr << "--- About to call importer->populateNet(net)" << std::endl;
    importer->populateNet(net);
	std::cerr << "--- After calling importer->populateNet(net)" << std::endl;
    importer.release();                     //We don't need importer anymore
    //! [Initialize network]

	// -----------------------------------------------------------------------
	// -------------------- Checking if Torch7 net was created ---------------
	// -----------------------------------------------------------------------
	if (net.empty())
	{
		std::cerr << "--- Torch7 net not created (no layers in net)" << std::endl;
		system("pause");
		exit(-1);
	}
	else
	{
		std::cerr << "-- Torch7 net has layers!!!!!!!!!!" << std::endl;
		//system("pause");
		//exit(-1);
	}

    //! [Prepare blob]
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		system("pause");
        exit(-1);
    }
	else
	{
		std::cerr << "-- Image file: " << imageFile << " read successfully." << endl;
	}

    resize(img, img, Size(224, 224));       //GoogLeNet accepts only 224x224 RGB-images
	// Next line has been modified, original code commented under it
	dnn::Blob inputBlob = dnn::Blob::fromImages(img);
    //dnn::Blob inputBlob = dnn::Blob(img);   //Convert Mat to dnn::Blob image batch

    //! [Prepare blob]

	// About to set blob
	std::cerr << "--- About to set blob." << endl;
    //! [Set input blob]
    net.setBlob(".data", inputBlob);        //set the network input
    //! [Set input blob]
	std::cerr << "-- Blob has been set!" << endl;

    //! [Make forward pass]
    net.forward();                          //compute output
    //! [Make forward pass]
	std::cerr << "Forward pass made." << endl;

    //! [Gather output]
    dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer

    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
    //! [Gather output]

    //! [Print results]
    std::vector<String> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    //! [Print results]

	system("pause");
    return 0;
} //main
