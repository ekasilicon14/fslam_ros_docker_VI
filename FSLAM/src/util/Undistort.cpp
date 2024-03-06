/**
* This file is part of DSO, written by Jakob Engel.
* It has been modified by Georges Younes, Daniel Asmar, John Zelek, and Yan Song Hu
*
* Copyright 2024 University of Waterloo and American University of Beirut.
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include <sstream>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <iterator>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/ImageRW.h"
#include "util/Undistort.h"

#include <algorithm>

namespace HSLAM
{

/**
 * @brief Construct a new Photometric Undistorter:: Photometric Undistorter object
 * 
 * Corrects the pixel brightness of the camera
 * A vignette image and gamma correction file are required
 * 
 * @param file 
 * @param noiseImage 
 * @param vignetteImage 
 * @param w_ 
 * @param h_ 
 */
PhotometricUndistorter::PhotometricUndistorter(
		std::string file,
		std::string noiseImage,
		std::string vignetteImage,
		Undistort* geomUndist,
		int w_, int h_)
{
	valid=false;
	vignetteMap=0;
	vignetteMapInv=0;
	w = w_;
	h = h_;
	// output = new ImageAndExposure(w,h);
	if(file=="" || vignetteImage=="")
	{
		printf("NO PHOTOMETRIC Calibration!\n");
	}


	// Read G
	std::ifstream f(file.c_str());
	printf("Reading Photometric Calibration from file %s\n",file.c_str());
	if (!f.good())
	{
		printf("PhotometricUndistorter: Could not open file!\n");
		return;
	}

	// Gamma Correction
	// Remaps luminance values to their correct values
	{
		std::string line;
		std::getline( f, line );
		std::istringstream l1i( line );
		std::vector<float> Gvec = std::vector<float>( std::istream_iterator<float>(l1i), std::istream_iterator<float>() );

        GDepth = Gvec.size();

        if(GDepth < 256)
        {
            printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected at least 256!\n",(int)Gvec.size());
            return;
        }


        for(int i=0;i<GDepth;i++) G[i] = Gvec[i];

        for(int i=0;i<GDepth-1;i++)
		{
			if(G[i+1] <= G[i])
			{
				printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
				return;
			}
		}

		float min=G[0];
        float max=G[GDepth-1];
        for(int i=0;i<GDepth;i++) G[i] = 255.0 * (G[i] - min) / (max-min);			// make it to 0..255 => 0..255.
	}

	// If no photometric calibration, set the gamma map to a linear correspondance
	if(setting_photometricCalibration==0)
	{
        for(int i=0;i<GDepth;i++) G[i]=255.0f*i/(float)(GDepth-1);
	}



	printf("Reading Vignette Image from %s\n",vignetteImage.c_str());
	MinimalImage<unsigned short>* vm16 = IOWrap::readImageBW_16U(vignetteImage.c_str());
	MinimalImageB* vm8 = IOWrap::readImageBW_8U(vignetteImage.c_str());
	
	// Vignette Correction
	// Removes the reduction of brightness near camera edges

	// First undistort the vignette map
	float* vignetteIn = nullptr;

	if (vm16 != 0)
		vignetteIn = geomUndist->geometricallyUnDistortVignette(vm16);
	else if (vm8 != 0)
		vignetteIn = geomUndist->geometricallyUnDistortVignette(vm8);
	else
	{
		printf("could not load Vignette Image from %s! exiting\n", vignetteImage.c_str());
		exit(1);
	}

	// Now create the vignette map
	vignetteMap = new float[w*h];
	vignetteMapInv = new float[w*h];

	if(vm16 != 0) // 16 bit vignette image
	{
		if(vm16->w != w ||vm16->h != h)
		{
			printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n",
					vm16->w, vm16->h, w, h);
			if(vm16!=0) delete vm16;
			if(vm8!=0) delete vm8;
			if(vignetteIn) delete[] vignetteIn;
			return;
		}

		float maxV=0;
		for(int i=0, size = w*h; i<size; ++i)
			if(vignetteIn[i] > maxV) maxV = vignetteIn[i];

		for(int i=0;i<w*h;i++)
		{
			vignetteMap[i] = vignetteIn[i] / maxV;
			vignetteMapInv[i] = 1.0f / vignetteMap[i];
		}
	}
	else if(vm8 != 0) // 8 bit vignette image
	{
		if(vm8->w != w ||vm8->h != h)
		{
			printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n",
					vm8->w, vm8->h, w, h);
			if(vm16!=0) delete vm16;
			if(vm8!=0) delete vm8;
			if(vignetteIn) delete[] vignetteIn;
			return;
		}

		float maxV=0;
		for(int i=0, size = w*h; i<size; ++i)
		if(vignetteIn[i] > maxV) maxV = vignetteIn[i];

		for(int i=0;i<w*h;i++)
		{
			vignetteMap[i] = vignetteIn[i] / maxV;
			vignetteMapInv[i] = 1.0f / vignetteMap[i];
		}
	}
	else
	{
		printf("PhotometricUndistorter: Invalid vignette image\n");
		if(vm16!=0) delete vm16;
		if(vm8!=0) delete vm8;
		if(vignetteIn) delete[] vignetteIn;
		return;
	}

	if(vm16!=0) delete vm16;
	if(vm8!=0) delete vm8;

	if(vignetteIn) delete[] vignetteIn;

	printf("Successfully read photometric calibration!\n");
	valid = true;
}

/**
 * @brief Destroy the Photometric Undistorter:: Photometric Undistorter object
 * 
 */
PhotometricUndistorter::~PhotometricUndistorter()
{
	if(vignetteMap != 0) delete[] vignetteMap;
	if(vignetteMapInv != 0) delete[] vignetteMapInv;
	//delete output;
}


/**
 * @brief Maps image to Gamma Correction with interpolation
 * 
 * @param image 
 */
void PhotometricUndistorter::unMapFloatImage(float* image)
{
	int wh=w*h;
	for(int i=0;i<wh;i++)
	{
		float BinvC;
		float color = image[i];

		if(color < 1e-3)
			BinvC=0.0f;
        else if(color > GDepth-1.01f)
            BinvC=GDepth-1.01f;
		else
		{
			int c = color;
			float a = color-c;
			BinvC=G[c]*(1-a) + G[c+1]*a; // Interpolate between G[c] and G[c+1]
		}

		float val = BinvC;
		if(val < 0) val = 0;
		image[i] = val;
	}
}

/**
 * @brief Photometrically undistorts image
 * 
 * Corrected output image is stored in class variable image (ImageAndExposure*)
 * 
 * @tparam T 
 * @param image_in 
 * @param exposure_time 
 * @param factor 
 */
void PhotometricUndistorter::processFrame(float* image_in, ImageAndExposure* output, float exposure_time, float factor, bool setMeta)
{
	int wh=w*h;
    float* data = output->image;
	assert(output->w == w && output->h == h);
	assert(data != 0);


	if(!valid || exposure_time <= 0 || setting_photometricCalibration==0) // Disable full photometric calibration.
	{
		for(int i=0; i<wh;i++)
		{
			// Multiply to factor and set to 0-255
			// This should divide by 256 if the image is 16 bit
			data[i] = factor*image_in[i];
		}
		if (setMeta){
			output->exposure_time = exposure_time;
			output->timestamp = 0;
		}
	}
	else
	{
		// Gamma Correction
		for(int i=0; i<wh;i++)
		{
			// Note that this also sets the image to 0-255
			data[i] = G[static_cast<unsigned char>(image_in[i])];
		}

		// Vignette Correction
		if(setting_photometricCalibration==2)
		{
			for(int i=0; i<wh;i++){
				data[i] *= vignetteMapInv[i];
				if (data[i] > 255) data[i] = 255;
			}
		}

		if (setMeta){
			output->exposure_time = exposure_time;
			output->timestamp = 0;
		}
	}


	if(!setting_useExposure && setMeta)
		output->exposure_time = 1;
}


/**
 * @brief Destroy the Undistort:: Undistort object
 * 
 */
Undistort::~Undistort()
{
	if(remapX != 0) delete[] remapX;
	if(remapY != 0) delete[] remapY;
}

/**
 * @brief Reads camera distortion file and loads data
 * 
 * @param configFilename 
 * @param gammaFilename 
 * @param vignetteFilename 
 * @return Undistort* 
 */
Undistort* Undistort::getUndistorterForFile(std::string configFilename, std::string gammaFilename, std::string vignetteFilename)
{
	printf("Reading Calibration from file %s",configFilename.c_str());

	std::ifstream f(configFilename.c_str());
	if (!f.good())
	{
		f.close();
		printf(" ... not found. Cannot operate without calibration, shutting down.\n");
		f.close();
		return 0;
	}

	printf(" ... found!\n");
	std::string l1;
	std::getline(f,l1);
	f.close();

	float ic[10];

	Undistort* u;

	// Default DSO assumed model from number of input parameters
    // For backwards-compatibility: Use RadTan model for 8 parameters.
	// fx, fy, cx, cy, k1, k2, r1, r2
	if(std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f",
			&ic[0], &ic[1], &ic[2], &ic[3],
			&ic[4], &ic[5], &ic[6], &ic[7]) == 8)
	{
        printf("found RadTan (OpenCV) camera model, building rectifier.\n");
        u = new UndistortRadTan(configFilename.c_str(), true);
		if(!u->isValid()) {delete u; return 0; }
    }

    // For backwards-compatibility: Use Pinhole / FoV model for 5 parameter.
	// fx, fy, cx, cy, scale
    else if(std::sscanf(l1.c_str(), "%f %f %f %f %f",
			&ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5)
	{
		if(ic[4]==0)
		{
			printf("found PINHOLE camera model, building rectifier.\n");
            u = new UndistortPinhole(configFilename.c_str(), true);
			if(!u->isValid()) {delete u; return 0; }
		}
		else
		{
			printf("found ATAN camera model, building rectifier.\n");
            u = new UndistortFOV(configFilename.c_str(), true);
			if(!u->isValid()) {delete u; return 0; }
		}
	}





    // clean model selection implementation.
	// fx, fy, cx, cy, k0, k1, k2, k3
    else if(std::sscanf(l1.c_str(), "KannalaBrandt %f %f %f %f %f %f %f %f",
            &ic[0], &ic[1], &ic[2], &ic[3],
            &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        u = new UndistortKB(configFilename.c_str(), false);
        if(!u->isValid()) {delete u; return 0; }
    }


	// fx, fy, cx, cy, k1, k2, r1, r2
    else if(std::sscanf(l1.c_str(), "RadTan %f %f %f %f %f %f %f %f",
            &ic[0], &ic[1], &ic[2], &ic[3],
            &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        u = new UndistortRadTan(configFilename.c_str(), false);
        if(!u->isValid()) {delete u; return 0; }
    }


	// fx, fy, cx, cy, k1, k2, k3, k4
    else if(std::sscanf(l1.c_str(), "EquiDistant %f %f %f %f %f %f %f %f",
            &ic[0], &ic[1], &ic[2], &ic[3],
            &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
    {
        u = new UndistortEquidistant(configFilename.c_str(), false);
        if(!u->isValid()) {delete u; return 0; }
    }


	// fx, fy, cx, cy, scale
    else if(std::sscanf(l1.c_str(), "FOV %f %f %f %f %f",
            &ic[0], &ic[1], &ic[2], &ic[3],
            &ic[4]) == 5)
    {
        u = new UndistortFOV(configFilename.c_str(), false);
        if(!u->isValid()) {delete u; return 0; }
    }


	// fx, fy, cx, cy
    else if(std::sscanf(l1.c_str(), "Pinhole %f %f %f %f %f",
            &ic[0], &ic[1], &ic[2], &ic[3],
            &ic[4]) == 5)
    {
        u = new UndistortPinhole(configFilename.c_str(), false);
        if(!u->isValid()) {delete u; return 0; }
    }


    else
    {
        printf("could not read calib file! exit.");
        exit(1);
    }

	// Load data for photometric calibration
	u->loadPhotometricCalibration(
				gammaFilename,
				"",
				vignetteFilename);

	return u;
}

/**
 * @brief Wrapper function to load photometric calibration
 * 
 * @param file 
 * @param noiseImage 
 * @param vignetteImage 
 */
void Undistort::loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage)
{
	photometricUndist = new PhotometricUndistorter(file, noiseImage, vignetteImage, this, w, h); //getOriginalSize()[0], getOriginalSize()[1]
}

/**
 * @brief Undistorts image
 * 
 * @tparam T 
 * @param image_raw 
 * @param exposure 
 * @param timestamp 
 * @param factor 
 * @return ImageAndExposure* 
 */
template<typename T>
float* Undistort::geometricallyUnDistortVignette(const MinimalImage<T>* image_raw)
{
	if(image_raw->w != wOrg || image_raw->h != hOrg)
	{
		printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d Exiting.\n",
					image_raw->w, image_raw->h, w, h);
		exit(1);
	}

	float* out_data = new float[w*h];
	T* in_data = image_raw->data;

	for(int idx = w*h-1;idx>=0;idx--)
	{
		// get interp. values
		float xx = remapX[idx];
		float yy = remapY[idx];

		if(xx<0)
			out_data[idx] = 0;
		else
		{
			// get integer and rational parts
			int xxi = xx;
			int yyi = yy;
			xx -= xxi;
			yy -= yyi;
			float xxyy = xx*yy;

			// get array base pointer
			const T* src = in_data + xxi + yyi * wOrg;

			// interpolate (bilinear)
			out_data[idx] =  xxyy * src[1+wOrg]
								+ (yy-xxyy) * src[wOrg]
								+ (xx-xxyy) * src[1]
								+ (1-xx-yy+xxyy) * src[0];
		}
	}
	return out_data;
}

float* Undistort::geometricallyUnDistortVignette(const MinimalImageB* image_raw)
{
	float* out_data = new float[w*h];
	unsigned char* in_data = image_raw->data;

	for(int idx = w*h-1;idx>=0;idx--)
	{
		// get interp. values
		float xx = remapX[idx];
		float yy = remapY[idx];

		if(xx<0)
			out_data[idx] = 0;
		else
		{
			// get integer and rational parts
			int xxi = xx;
			int yyi = yy;
			xx -= xxi;
			yy -= yyi;
			float xxyy = xx*yy;

			// get array base pointer
			const unsigned char* src = in_data + xxi + yyi * wOrg;

			// interpolate (bilinear)
			out_data[idx] =  xxyy * src[1+wOrg]
								+ (yy-xxyy) * src[wOrg]
								+ (xx-xxyy) * src[1]
								+ (1-xx-yy+xxyy) * src[0];
		}
	}

	return out_data;
}


template<typename T>
ImageAndExposure* Undistort::undistort(const MinimalImage<T>* image_raw, float exposure, double timestamp, float factor, bool useColourPassed) const
{
	if(image_raw->w != wOrg || image_raw->h != hOrg)
	{
		printf("Undistort::undistort: wrong image size (%d %d instead of %d %d) \n", image_raw->w, image_raw->h, w, h);
		exit(1);
	}

	ImageAndExposure* result = new ImageAndExposure(w, h, timestamp, useColourPassed);


	if (!passthrough) // If all inputs are valid
	{
		float* out_data = result->PhoUncalibImage;
		T *in_data = image_raw->data;

		float* noiseMapX=0;
		float* noiseMapY=0;
		if(benchmark_varNoise>0) // Add noise to the images for testing
		{
			// Generate grid of random noise
			int numnoise=(benchmark_noiseGridsize+8)*(benchmark_noiseGridsize+8);
			noiseMapX=new float[numnoise];
			noiseMapY=new float[numnoise];
			memset(noiseMapX,0,sizeof(float)*numnoise);
			memset(noiseMapY,0,sizeof(float)*numnoise);

			for(int i=0;i<numnoise;i++)
			{
				// Random values are between (-benchmark_varNoise, benchmark_varNoise)
				noiseMapX[i] =  benchmark_varNoise * 2 * (rand()/(float)RAND_MAX - 0.5f);
				noiseMapY[i] =  benchmark_varNoise * 2 * (rand()/(float)RAND_MAX - 0.5f);
			}
		}


		for(int idx = w*h-1;idx>=0;idx--)
		{
			// Remap pixels to undistorted positions
			float xx = remapX[idx];
			float yy = remapY[idx];



			if(benchmark_varNoise>0) // Add remapping noise to the images for experiments
			{
				float deltax = getInterpolatedElement11BiCub(noiseMapX, 4+(xx/(float)wOrg)*benchmark_noiseGridsize, 4+(yy/(float)hOrg)*benchmark_noiseGridsize, benchmark_noiseGridsize+8 );
				float deltay = getInterpolatedElement11BiCub(noiseMapY, 4+(xx/(float)wOrg)*benchmark_noiseGridsize, 4+(yy/(float)hOrg)*benchmark_noiseGridsize, benchmark_noiseGridsize+8 );
				float x = idx%w + deltax;
				float y = idx/w + deltay;
				if(x < 0.01) x = 0.01;
				if(y < 0.01) y = 0.01;
				if(x > w-1.01) x = w-1.01;
				if(y > h-1.01) y = h-1.01;

				xx = getInterpolatedElement(remapX, x, y, w);
				yy = getInterpolatedElement(remapY, x, y, w);
			}


			if(xx<0)
				out_data[idx] = 0;
			else
			{
				// Get integer and rational parts
				int xxi = xx;
				int yyi = yy;
				xx -= xxi;
				yy -= yyi;
				float xxyy = xx*yy;

				// get array base pointer
				const T* src = in_data + xxi + yyi * wOrg;

				// Interpolate (bilinear)
				out_data[idx] =  xxyy * src[1+wOrg]
									+ (yy-xxyy) * src[wOrg]
									+ (xx-xxyy) * src[1]
									+ (1-xx-yy+xxyy) * src[0];
			}
		}

		if(benchmark_varNoise>0)
		{
			delete[] noiseMapX;
			delete[] noiseMapY;
		}

	}
	else
	{
		for (int i = 0, size = w * h; i < size; ++i)
			result->PhoUncalibImage[i] = image_raw->data[i];
	}

	photometricUndist->processFrame(result->PhoUncalibImage, result, exposure, factor, true);

	applyBlurNoise(result->image);

	return result;
}
template ImageAndExposure* Undistort::undistort<unsigned char>(const MinimalImage<unsigned char>* image_raw, float exposure, double timestamp, float factor, bool useColourPassed) const;
template ImageAndExposure* Undistort::undistort<unsigned short>(const MinimalImage<unsigned short>* image_raw, float exposure, double timestamp, float factor, bool useColourPassed) const;

template<typename T>
void Undistort::undistort_colour(MinimalImage<T>* r_image, MinimalImage<T>* g_image, MinimalImage<T>* b_image, ImageAndExposure* out_image, float exposure, double timestamp, float factor)
{
	MinimalImage<T>* channels[3];
	channels[0] = r_image;
	channels[1] = g_image;
	channels[2] = b_image;

	float* out_channels[3];
	out_channels[0] = out_image->r_image;
	out_channels[1] = out_image->g_image;
	out_channels[2] = out_image->b_image;

	if(out_image->useColour == false){
		printf("Colour not actually being used\n");
		exit(1);
	}

	for (unsigned short i = 0; i<3; i++){
		if(channels[i]->w != wOrg || channels[i]->h != hOrg)
		{
			printf("Undistort::undistort: wrong colour image size (%d %d instead of %d %d) \n", channels[i]->w, channels[i]->h, w, h);
			exit(1);
		}
	}

	for (unsigned short i = 0; i<3; i++){
		if (!passthrough) // If all inputs are valid
		{
			float* out_data = out_channels[i];
			T* in_data = channels[i]->data;

			for(int idx = w*h-1;idx>=0;idx--)
			{
				// Remap pixels to undistorted positions
				float xx = remapX[idx];
				float yy = remapY[idx];
				if(xx<0){
					out_data[idx] = 0;
				}
				else
				{
					// Get integer and rational parts
					int xxi = xx;
					int yyi = yy;
					xx -= xxi;
					yy -= yyi;
					float xxyy = xx*yy;
					// Get array base pointer
					const T* src = in_data + xxi + yyi * wOrg;
					// Interpolate (bilinear)
					out_data[idx] =  xxyy * src[1+wOrg]
										+ (yy-xxyy) * src[wOrg]
										+ (xx-xxyy) * src[1]
										+ (1-xx-yy+xxyy) * src[0];
				}
			}
		}
		else
		{
			for (int j = 0, size = w * h; j < size; ++j)
				out_channels[i][j] = channels[i]->data[i];
		}
	}
}
template void Undistort::undistort_colour<unsigned char>(MinimalImage<unsigned char>* r_image, MinimalImage<unsigned char>* g_image, MinimalImage<unsigned char>* b_image, ImageAndExposure* out_image, float exposure, double timestamp, float factor=1);
template void Undistort::undistort_colour<unsigned short>(MinimalImage<unsigned short>* r_image, MinimalImage<unsigned short>* g_image, MinimalImage<unsigned short>* b_image, ImageAndExposure* out_image, float exposure, double timestamp, float factor=1);


/**
 * @brief Applies gaussian blur noise for experiments
 * 
 * @param img 
 */
void Undistort::applyBlurNoise(float* img) const
{
	if(benchmark_varBlurNoise==0) return;

	int numnoise=(benchmark_noiseGridsize+8)*(benchmark_noiseGridsize+8);
	float* noiseMapX=new float[numnoise];
	float* noiseMapY=new float[numnoise];
	float* blutTmp=new float[w*h];

	if(benchmark_varBlurNoise>0)
	{
		for(int i=0;i<numnoise;i++)
		{
				noiseMapX[i] =  benchmark_varBlurNoise  * (rand()/(float)RAND_MAX);
				noiseMapY[i] =  benchmark_varBlurNoise  * (rand()/(float)RAND_MAX);
		}
	}


	float gaussMap[1000];
	for(int i=0;i<1000;i++)
		gaussMap[i] = expf((float)(-i*i/(100.0*100.0)));

	// Applies randomized Gaussian kernel to image
	// x-blur.
	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			float xBlur = getInterpolatedElement11BiCub(noiseMapX,
					4+(x/(float)w)*benchmark_noiseGridsize,
					4+(y/(float)h)*benchmark_noiseGridsize,
					benchmark_noiseGridsize+8 );

			if(xBlur < 0.01) xBlur=0.01;


			int kernelSize = 1 + (int)(1.0f+xBlur*1.5);
			float sumW=0;
			float sumCW=0;
			for(int dx=0; dx <= kernelSize; dx++)
			{
				int gmid = 100.0f*dx/xBlur + 0.5f;
				if(gmid > 900 ) gmid = 900;
				float gw = gaussMap[gmid];

				if(x+dx>0 && x+dx<w)
				{
					sumW += gw;
					sumCW += gw * img[x+dx+y*this->w];
				}

				if(x-dx>0 && x-dx<w && dx!=0)
				{
					sumW += gw;
					sumCW += gw * img[x-dx+y*this->w];
				}
			}

			blutTmp[x+y*this->w] = sumCW / sumW;
		}

	// y-blur.
	for(int x=0;x<w;x++)
		for(int y=0;y<h;y++)
		{
			float yBlur = getInterpolatedElement11BiCub(noiseMapY,
					4+(x/(float)w)*benchmark_noiseGridsize,
					4+(y/(float)h)*benchmark_noiseGridsize,
					benchmark_noiseGridsize+8 );

			if(yBlur < 0.01) yBlur=0.01;

			int kernelSize = 1 + (int)(0.9f+yBlur*2.5);
			float sumW=0;
			float sumCW=0;
			for(int dy=0; dy <= kernelSize; dy++)
			{
				int gmid = 100.0f*dy/yBlur + 0.5f;
				if(gmid > 900 ) gmid = 900;
				float gw = gaussMap[gmid];

				if(y+dy>0 && y+dy<h)
				{
					sumW += gw;
					sumCW += gw * blutTmp[x+(y+dy)*this->w];
				}

				if(y-dy>0 && y-dy<h && dy!=0)
				{
					sumW += gw;
					sumCW += gw * blutTmp[x+(y-dy)*this->w];
				}
			}
			img[x+y*this->w] = sumCW / sumW;
		}



	delete[] noiseMapX;
	delete[] noiseMapY;
}

/**
 * @brief Finds optimal crop for image
 * 
 * Sets crop to max size and the shrinks it till there are no more invalid pixels
 * 
 */
void Undistort::makeOptimalK_crop()
{
	printf("finding CROP optimal new model!\n");
	K.setIdentity();

	// 1. Stretch the center lines as far as possible, to get initial coarse crop
	// This is done by running the undistortion on a horizontal and vertical line of length -0.5 to 0.5
	float* tgX = new float[100000];
	float* tgY = new float[100000];
	float minX = 0;
	float maxX = 0;
	float minY = 0;
	float maxY = 0;

	for(int x=0; x<100000;x++) {tgX[x] = (x-50000.0f) / 10000.0f; tgY[x] = 0;}
	distortCoordinates(tgX, tgY,tgX, tgY,100000);
	for(int x=0; x<100000;x++)
	{
		if(tgX[x] > 0 && tgX[x] < wOrg-1)
		{
			if(minX==0) minX = (x-50000.0f) / 10000.0f;
			maxX = (x-50000.0f) / 10000.0f;
		}
	}
	for(int y=0; y<100000;y++) {tgY[y] = (y-50000.0f) / 10000.0f; tgX[y] = 0;}
	distortCoordinates(tgX, tgY,tgX, tgY,100000);
	for(int y=0; y<100000;y++)
	{
		if(tgY[y] > 0 && tgY[y] < hOrg-1)
		{
			if(minY==0) minY = (y-50000.0f) / 10000.0f;
			maxY = (y-50000.0f) / 10000.0f;
		}
	}
	delete[] tgX;
	delete[] tgY;

	minX *= 1.01;
	maxX *= 1.01;
	minY *= 1.01;
	maxY *= 1.01;



	printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);



	// 2. While there are invalid pixels at the border: shrink square at the side that has invalid pixels,
	// If several to choose from, shrink the wider dimension.
	bool oobLeft=true, oobRight=true, oobTop=true, oobBottom=true;
	int iteration=0;
	while(oobLeft || oobRight || oobTop || oobBottom)
	{
		oobLeft=oobRight=oobTop=oobBottom=false;
		for(int y=0;y<h;y++)
		{
			remapX[y*2] = minX;
			remapX[y*2+1] = maxX;
			remapY[y*2] = remapY[y*2+1] = minY + (maxY-minY) * (float)y / ((float)h-1.0f);
		}
		distortCoordinates(remapX, remapY,remapX, remapY,2*h);
		for(int y=0;y<h;y++)
		{
			if(!(remapX[2*y] > 0 && remapX[2*y] < wOrg-1))
				oobLeft = true;
			if(!(remapX[2*y+1] > 0 && remapX[2*y+1] < wOrg-1))
				oobRight = true;
		}



		for(int x=0;x<w;x++)
		{
			remapY[x*2] = minY;
			remapY[x*2+1] = maxY;
			remapX[x*2] = remapX[x*2+1] = minX + (maxX-minX) * (float)x / ((float)w-1.0f);
		}
		distortCoordinates(remapX, remapY,remapX, remapY,2*w);


		for(int x=0;x<w;x++)
		{
			if(!(remapY[2*x] > 0 && remapY[2*x] < hOrg-1))
				oobTop = true;
			if(!(remapY[2*x+1] > 0 && remapY[2*x+1] < hOrg-1))
				oobBottom = true;
		}


		if((oobLeft || oobRight) && (oobTop || oobBottom))
		{
			if((maxX-minX) > (maxY-minY))
				oobBottom = oobTop = false;	// only shrink left/right
			else
				oobLeft = oobRight = false; // only shrink top/bottom
		}

		if(oobLeft) minX *= 0.995;
		if(oobRight) maxX *= 0.995;
		if(oobTop) minY *= 0.995;
		if(oobBottom) maxY *= 0.995;

		iteration++;


		printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration,  minX, maxX, minY, maxY);
		if(iteration > 500)
		{
			printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
			exit(1);
		}
	}

	K(0,0) = ((float)w-1.0f)/(maxX-minX);
	K(1,1) = ((float)h-1.0f)/(maxY-minY);
	K(0,2) = -minX*K(0,0);
	K(1,2) = -minY*K(1,1);

}

void Undistort::makeOptimalK_full()
{
	// TODO
	assert(false);
}

/**
 * @brief Parses inputted data into a remapping array
 * 
 * Used by the initializer of the distortion model classes
 * Function changes depending on distortion model
 * Camera text file is formated like this:
 * line 1: Undistortion type and camera parameters
 * line 2: Input image size
 * line 3: Crop settings or output camera parameteers
 * line 4: Output image size
 * 
 * @param configFileName 
 * @param nPars 
 * @param prefix 
 */
void Undistort::readFromFile(const char* configFileName, int nPars, std::string prefix)
{
	photometricUndist=0;
	valid = false;
	passthrough=false;
	remapX = 0;
	remapY = 0;
	
	float outputCalibration[5];

	parsOrg = VecX(nPars);

	// read parameters
	std::ifstream infile(configFileName);
	assert(infile.good());

    std::string l1,l2,l3,l4;

	std::getline(infile,l1);
	std::getline(infile,l2);
    std::getline(infile,l3);
    std::getline(infile,l4);

    // line 1 & line 2
    if(nPars == 5) // fov model
	{
		char buf[1000];
		snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf", prefix.c_str());

		if(std::sscanf(l1.c_str(), buf, &parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4]) == 5 &&
				std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2)
		{
			printf("Input resolution: %d %d\n",wOrg, hOrg);
			printf("In: %f %f %f %f %f\n",
					parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3], parsOrg[4]);
		}
		else
		{
			printf("Failed to read camera calibration (invalid format?)\nCalibration file: %s\n", configFileName);
			infile.close();
			return;
		}
	}
	else if(nPars == 8) // KB, equi & radtan model
	{
		char buf[1000];
		snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf", prefix.c_str());

		if(std::sscanf(l1.c_str(), buf,
				&parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4],
				&parsOrg[5], &parsOrg[6], &parsOrg[7]) == 8 &&
				std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2)
		{
			printf("Input resolution: %d %d\n",wOrg, hOrg);
			printf("In: %s%f %f %f %f %f %f %f %f\n",
					prefix.c_str(),
					parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3], parsOrg[4], parsOrg[5], parsOrg[6], parsOrg[7]);
		}
		else
		{
			printf("Failed to read camera calibration (invalid format?)\nCalibration file: %s\n", configFileName);
			infile.close();
			return;
		}
	}
	else
	{
		printf("called with invalid number of parameters.... forgot to implement me?\n");
		infile.close();
		return;
	}



	// Change relative width and height to absolute size
    if(parsOrg[2] < 1 && parsOrg[3] < 1)
    {
        printf("\n\nFound fx=%f, fy=%f, cx=%f, cy=%f.\n I'm assuming this is the \"relative\" calibration file format,"
               "and will rescale this by image width / height to fx=%f, fy=%f, cx=%f, cy=%f.\n\n",
               parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3],
               parsOrg[0] * wOrg, parsOrg[1] * hOrg, parsOrg[2] * wOrg - 0.5, parsOrg[3] * hOrg - 0.5 );

        // rescale and substract 0.5 offset.
        // the 0.5 is because it is assumed the calibration is given such that the pixel at (0,0)
        // contains the integral over intensity over [0,0]-[1,1], whereas I assume the pixel (0,0)
        // to contain a sample of the intensity ot [0,0], which is best approximated by the integral over
        // [-0.5,-0.5]-[0.5,0.5]. Thus, the shift by -0.5.
        parsOrg[0] = parsOrg[0] * wOrg;
        parsOrg[1] = parsOrg[1] * hOrg;
        parsOrg[2] = parsOrg[2] * wOrg - 0.5;
        parsOrg[3] = parsOrg[3] * hOrg - 0.5;
    }



	// line 3
	if(l3 == "crop")
	{
		outputCalibration[0] = -1;
        printf("Out: Rectify Crop\n");
	}
	else if(l3 == "full")
	{
		outputCalibration[0] = -2;
        printf("Out: Rectify Full\n");
	}
	else if(l3 == "none")
	{
		outputCalibration[0] = -3;
        printf("Out: No Rectification\n");
	}
	else if(std::sscanf(l3.c_str(), "%f %f %f %f %f", &outputCalibration[0], &outputCalibration[1], &outputCalibration[2], &outputCalibration[3], &outputCalibration[4]) == 5)
	{
		printf("Out: %f %f %f %f %f\n",
				outputCalibration[0], outputCalibration[1], outputCalibration[2], outputCalibration[3], outputCalibration[4]);

	}
	else
	{
		printf("Out: Failed to Read Output pars... not rectifying.\n");
		infile.close();
		return;
	}


	// line 4
	if(std::sscanf(l4.c_str(), "%d %d", &w, &h) == 2)
	{
		if(benchmarkSetting_width != 0) // Experimental
        {
			w = benchmarkSetting_width;
            if(outputCalibration[0] == -3)
                outputCalibration[0] = -1;  // crop instead of none, since probably resolution changed.
        }
        if(benchmarkSetting_height != 0) // Experimental
        {
			h = benchmarkSetting_height;
            if(outputCalibration[0] == -3)
                outputCalibration[0] = -1;  // crop instead of none, since probably resolution changed.
        }

		printf("Output resolution: %d %d\n",w, h);
	}
	else
	{
		printf("Out: Failed to Read Output resolution... not rectifying.\n");
		valid = false;
    }

    remapX = new float[w*h];
    remapY = new float[w*h];

	// Cropping sets the output camera K matrix
	if(outputCalibration[0] == -1) // Crop
		makeOptimalK_crop();
	else if(outputCalibration[0] == -2) // Crop full
		makeOptimalK_full();
	else if(outputCalibration[0] == -3) // None
	{
		if(w != wOrg || h != hOrg)
		{
			printf("ERROR: rectification mode none requires input and output dimenstions to match!\n\n");
			exit(1);
		}
		// Set K to be the same as input
		K.setIdentity();
        K(0,0) = parsOrg[0];
        K(1,1) = parsOrg[1];
        K(0,2) = parsOrg[2];
        K(1,2) = parsOrg[3];
		passthrough = true;
	}
	else // Inputted value
	{


        if(outputCalibration[2] > 1 || outputCalibration[3] > 1)
        {
            printf("\n\n\nWARNING: given output calibration (%f %f %f %f) seems wrong. It needs to be relative to image width / height!\n\n\n",
                   outputCalibration[0],outputCalibration[1],outputCalibration[2],outputCalibration[3]);
        }


		// Set K to be given parameters
		K.setIdentity();
        K(0,0) = outputCalibration[0] * w;
        K(1,1) = outputCalibration[1] * h;
        K(0,2) = outputCalibration[2] * w - 0.5;
        K(1,2) = outputCalibration[3] * h - 0.5;
	}

	if(benchmarkSetting_fxfyfac != 0) // Experimental
	{
		K(0,0) = fmax(benchmarkSetting_fxfyfac, (float)K(0,0));
		K(1,1) = fmax(benchmarkSetting_fxfyfac, (float)K(1,1));
        passthrough = false; // cannot pass through when fx / fy have been overwritten.
	}


	// Set remapping arrays
	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			remapX[x+y*w] = x;
			remapY[x+y*w] = y;
		}

	// Create remapping matrix
	distortCoordinates(remapX, remapY, remapX, remapY, h*w);


	// Make rounding resistant.
	for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			// make rounding resistant.
			float ix = remapX[x+y*w];
			float iy = remapY[x+y*w];

			if(ix == 0) ix = 0.001;
			if(iy == 0) iy = 0.001;
			if(ix == wOrg-1) ix = wOrg-1.001;
			if(iy == hOrg-1) ix = hOrg-1.001;

			if(ix > 0 && iy > 0 && ix < wOrg-1 &&  iy < hOrg-1)
			{
				remapX[x+y*w] = ix;
				remapY[x+y*w] = iy;
			}
			else
			{
				remapX[x+y*w] = -1;
				remapY[x+y*w] = -1;
			}
		}

	valid = true;




	printf("\nRectified Camera Matrix:\n");
	std::cout << K << "\n\n";

}


// The functions below implement the distortCoordinates for the respective camera types
UndistortFOV::UndistortFOV(const char* configFileName, bool noprefix)
{
    printf("Creating FOV undistorter\n");

    if(noprefix)
        readFromFile(configFileName, 5);
    else
        readFromFile(configFileName, 5, "FOV ");
}
UndistortFOV::~UndistortFOV()
{
}

void UndistortFOV::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const
{
	float dist = parsOrg[4];
	float d2t = 2.0f * tan(dist / 2.0f);



	// current camera parameters
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];



	float ofx = K(0,0);
	float ofy = K(1,1);
	float ocx = K(0,2);
	float ocy = K(1,2);

	for(int i=0;i<n;i++)
	{
		float x = in_x[i];
		float y = in_y[i];
		float ix = (x - ocx) / ofx;
		float iy = (y - ocy) / ofy;

		float r = sqrtf(ix*ix + iy*iy);
		float fac = (r==0 || dist==0) ? 1 : atanf(r * d2t)/(dist*r);

		ix = fx*fac*ix+cx;
		iy = fy*fac*iy+cy;

		out_x[i] = ix;
		out_y[i] = iy;
	}
}






UndistortRadTan::UndistortRadTan(const char* configFileName, bool noprefix)
{
    printf("Creating RadTan undistorter\n");

    if(noprefix)
        readFromFile(configFileName, 8);
    else
        readFromFile(configFileName, 8,"RadTan ");
}
UndistortRadTan::~UndistortRadTan()
{
}

void UndistortRadTan::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const
{
    // RADTAN
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];
    float k1 = parsOrg[4];
    float k2 = parsOrg[5];
    float r1 = parsOrg[6];
    float r2 = parsOrg[7];

    float ofx = K(0,0);
    float ofy = K(1,1);
    float ocx = K(0,2);
    float ocy = K(1,2);



    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];

        // RADTAN
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        float mx2_u = ix * ix;
        float my2_u = iy * iy;
        float mxy_u = ix * iy;
        float rho2_u = mx2_u+my2_u;
        float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        float x_dist = ix + ix * rad_dist_u + 2.0 * r1 * mxy_u + r2 * (rho2_u + 2.0 * mx2_u);
        float y_dist = iy + iy * rad_dist_u + 2.0 * r2 * mxy_u + r1 * (rho2_u + 2.0 * my2_u);
        float ox = fx*x_dist+cx;
        float oy = fy*y_dist+cy;


        out_x[i] = ox;
        out_y[i] = oy;
    }


}



UndistortEquidistant::UndistortEquidistant(const char* configFileName, bool noprefix)
{
    printf("Creating Equidistant undistorter\n");

    if(noprefix)
        readFromFile(configFileName, 8);
    else
        readFromFile(configFileName, 8,"EquiDistant ");
}
UndistortEquidistant::~UndistortEquidistant()
{
}

void UndistortEquidistant::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const
{
    // EQUI
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];
    float k1 = parsOrg[4];
    float k2 = parsOrg[5];
    float k3 = parsOrg[6];
    float k4 = parsOrg[7];


    float ofx = K(0,0);
    float ofy = K(1,1);
    float ocx = K(0,2);
    float ocy = K(1,2);



    for(int i=0;i<n;i++)
    {
        float x = in_x[i];
        float y = in_y[i];

        // EQUI
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        float r = sqrt(ix * ix + iy * iy);
        float theta = atan(r);
        float theta2 = theta * theta;
        float theta4 = theta2 * theta2;
        float theta6 = theta4 * theta2;
        float theta8 = theta4 * theta4;
        float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        float scaling = (r > 1e-8) ? thetad / r : 1.0;
        float ox = fx*ix*scaling + cx;
        float oy = fy*iy*scaling + cy;

        out_x[i] = ox;
        out_y[i] = oy;
    }
}



UndistortKB::UndistortKB(const char* configFileName, bool noprefix)
{
	printf("Creating KannalaBrandt undistorter\n");

    if(noprefix)
        readFromFile(configFileName, 8);
    else
        readFromFile(configFileName, 8,"KannalaBrandt ");
}
UndistortKB::~UndistortKB()
{
}

void UndistortKB::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const
{
    const float fx = parsOrg[0];
	const float fy = parsOrg[1];
	const float cx = parsOrg[2];
	const float cy = parsOrg[3];
    const float k0 = parsOrg[4];
    const float k1 = parsOrg[5];
    const float k2 = parsOrg[6];
    const float k3 = parsOrg[7];

    const float ofx = K(0,0);
    const float ofy = K(1,1);
    const float ocx = K(0,2);
    const float ocy = K(1,2);


	for(int i=0;i<n;i++)
	{
		float x = in_x[i];
		float y = in_y[i];

		// RADTAN
		float ix = (x - ocx) / ofx;
		float iy = (y - ocy) / ofy;

	    const float Xsq_plus_Ysq = ix*ix + iy*iy;
	    const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
	    const float theta = atan2f( sqrt_Xsq_Ysq, 1 );
	    const float theta2 = theta*theta;
	    const float theta3 = theta2*theta;
	    const float theta5 = theta3*theta2;
	    const float theta7 = theta5*theta2;
	    const float theta9 = theta7*theta2;
	    const float r = theta + k0*theta3 + k1*theta5 + k2*theta7 + k3*theta9;

	    if(sqrt_Xsq_Ysq < 1e-6)
	    {
	    	out_x[i] = fx * ix + cx;
	    	out_y[i] = fy * iy + cy;
	    }
	    else
	    {
	    	out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
	    	out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
	    }
	}
}





UndistortPinhole::UndistortPinhole(const char* configFileName, bool noprefix)
{
    if(noprefix)
        readFromFile(configFileName, 5);
    else
        readFromFile(configFileName, 5,"Pinhole ");

}
UndistortPinhole::~UndistortPinhole()
{
}

void UndistortPinhole::distortCoordinates(float* in_x, float* in_y, float* out_x, float* out_y, int n) const
{
	// current camera parameters
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];

	float ofx = K(0,0);
	float ofy = K(1,1);
	float ocx = K(0,2);
	float ocy = K(1,2);

	for(int i=0;i<n;i++)
	{
		float x = in_x[i];
		float y = in_y[i];
		float ix = (x - ocx) / ofx;
		float iy = (y - ocy) / ofy;
		ix = fx*ix+cx;
		iy = fy*iy+cy;
		out_x[i] = ix;
		out_y[i] = iy;
	}
}


}
