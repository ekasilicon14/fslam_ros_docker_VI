/**
* This file is part of DSO.
* 
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


#pragma once
#include <cstring>
#include <iostream>


namespace HSLAM
{


class ImageAndExposure
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	float* image;			// irradiance. between 0 and 256
	float *PhoUncalibImage;

	float* r_image;
	float* g_image;
	float* b_image;
	bool useColour;

	int w, h; // width and height;
	double timestamp;
	float exposure_time;	// exposure time in ms.
	inline ImageAndExposure(int w_, int h_, double timestamp_=0, bool useColourPassed = false) : w(w_), h(h_), timestamp(timestamp_)
	{
		image = new float[w*h];
		PhoUncalibImage = new float[w * h];

		useColour = useColourPassed;

		if (useColour){
			r_image = new float[w*h];
			g_image = new float[w*h];
			b_image = new float[w*h];
		} else {
			r_image = nullptr;
			g_image = nullptr;
			b_image = nullptr;	
		}
		exposure_time = 1;
	}
	inline ~ImageAndExposure()
	{
		delete[] image;
		delete[] PhoUncalibImage;

		if (useColour){
			delete[] r_image;
			delete[] g_image;
			delete[] b_image;
		}
	}

	inline void copyMetaTo(ImageAndExposure &other)
	{
		other.exposure_time = exposure_time;
	}

	inline ImageAndExposure* getDeepCopy()
	{
		ImageAndExposure* img = new ImageAndExposure(w,h,timestamp,useColour);
		img->exposure_time = exposure_time;
		memcpy(img->image, image, w*h*sizeof(float));

		if (useColour){
			img->useColour = useColour;
			memcpy(img->r_image, r_image, w*h*sizeof(float));
			memcpy(img->g_image, g_image, w*h*sizeof(float));
			memcpy(img->b_image, b_image, w*h*sizeof(float));
		}

		return img;
	}
};


}
