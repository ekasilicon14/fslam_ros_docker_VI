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



#pragma once
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"

#if HAS_ZIPLIB
	#include "zip.h"
#endif

#include <boost/thread.hpp>

using namespace HSLAM;



/**
 * @brief Gets the files from the given directory in alphabetical order
 * 
 * @param dir 		Input directory
 * @param files 	Vector of file paths
 * @return int 		-1 if funaction failed. Otherwise, return number of files
 */
inline int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);


	// files are sorted alphabetically
    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(unsigned int i=0;i<files.size();i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}


struct PrepImageItem
{
	int id;
	bool isQueud;
	ImageAndExposure* pt;

	inline PrepImageItem(int _id)
	{
		id=_id;
		isQueud = false;
		pt=0;
	}

	inline void release()
	{
		if(pt!=0) delete pt;
		pt=0;
	}
};




/**
 * @brief Handles the inputting of images into the full system
 * 
 */
class ImageFolderReader
{
public:
	/**
	 * @brief Construct a new Image Folder Reader object
	 * 
	 * @param path 				Path to video files
	 * @param calibFile 		Path to calibration file
	 * @param gammaFile 		Path to photometric gamma correction file
	 * @param vignetteFile 		Path to photometric vignette correction file
	 * @param use16BitPassed 	16 or 8 bit images
	 */
	ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile, bool use16BitPassed, bool useColourPassed)
	{
		this->path = path;
		this->calibfile = calibFile;
		use16Bit = use16BitPassed;
		useColour = useColourPassed;

#if HAS_ZIPLIB
		ziparchive=0;
		databuffer=0;
#endif

		isZipped = (path.length()>4 && path.substr(path.length()-4) == ".zip");





		if(isZipped)
		{
#if HAS_ZIPLIB
			int ziperror=0;
			ziparchive = zip_open(path.c_str(),  ZIP_RDONLY, &ziperror);
			if(ziperror!=0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
				exit(1);
			}

			files.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			for(int k=0;k<numEntries;k++)
			{
				const char* name = zip_get_name(ziparchive, k,  ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if(nstr == "." || nstr == "..") continue;
				files.push_back(name);
			}

			printf("got %d entries and %d files!\n", numEntries, (int)files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
		else
			getdir (path, files);


		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);


		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width=undistort->getSize()[0];
		height=undistort->getSize()[1];


		// Load timestamps if possible.
		loadTimestamps();
		printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());

	}
	/**
	 * @brief Destroy the Image Folder Reader object
	 * 
	 */
	~ImageFolderReader()
	{
#if HAS_ZIPLIB
		if(ziparchive!=0) zip_close(ziparchive);
		if(databuffer!=0) delete databuffer;
#endif


		delete undistort;
	};

	/**
	 * @brief Get the Original Calib object
	 * 
	 * @return Eigen::VectorXf 
	 */
	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	/**
	 * @brief Get the Original Dimensions object
	 * 
	 * @return Eigen::Vector2i 
	 */
	Eigen::Vector2i getOriginalDimensions()
	{
		return  undistort->getOriginalSize();
	}

	/**
	 * @brief Get the Calib Mono object
	 * 
	 * @param K 	K matrix of camera
	 * @param w 	width
	 * @param h 	height
	 */
	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	/**
	 * @brief Set the Global Calibration object
	 * 
	 * Wrapper for getCalibMono and setGlobalCalib
	 * 
	 */
	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
	}

	/**
	 * @brief Get the number of images
	 * 
	 * @return int 
	 */
	int getNumImages()
	{
		return files.size();
	}

	int get_undist_width()
	{
		return undistort->getSize()[0];
	}

	int get_undist_height()
	{
		return undistort->getSize()[1];
	}

	/**
	 * @brief Get the timestamp of the frame with the id
	 * 
	 * @param id 
	 * @return double 
	 */
	double getTimestamp(int id)
	{
		if(timestamps.size()==0) return id*0.04f; //0.1f = 10 Hz
		if(id >= (int)timestamps.size()) return 0;
		if(id < 0) return 0;
		return timestamps[id];
	}

	/**
	 * @brief Get the Filename object
	 * 
	 * @param id 
	 * @return std::string 
	 */
    std::string getFilename(int id)
    {
        return files[id];
    }

	void prepImage(int id, bool as8U=false)
	{

	}


	/**
	 * @brief Wrapper function for getImageRaw_internal
	 * 
	 * Returns as MinimalImage<unsigned char> pointer
	 * 
	 * @param id 
	 * @return MinimalImageB* 
	 */
	MinimalImageB* getImageRaw(int id)
	{
			return getImageRaw_internal(id,0);
	}

	/**
	 * @brief Wrapper function for getImageRaw_internal
	 * 
	 * Returns as ImageAndExposure pointer
	 * 
	 * @param id 
	 * @param forceLoadDirectly 
	 * @return ImageAndExposure* 
	 */
	ImageAndExposure* getImage(int id, bool forceLoadDirectly=false)
	{
		return getImage_internal(id, 0);
	}


	/**
	 * @brief Get the photometric gamma array
	 * 
	 * @return float* 
	 */
	inline float* getPhotometricGamma()
	{
		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}


	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort* undistort;
private:

	/**
	 * @brief Reads image
	 * 
	 * Either uses the provided image reader from the IOWrapper or extracts a zip file
	 * No timestamp or exposure is provided
	 * 
	 * @param id 
	 * @param unused 
	 * @return MinimalImageB* 
	 */
	MinimalImageB* getImageRaw_internal(int id, int unused)
	{
	    assert(!use16Bit);
		if(!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(files[id]);
		}
		else
		{
#if HAS_ZIPLIB
			if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
			zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

			if(readbytes > (long)widthOrg*heightOrg*6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg*heightOrg*30];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*30+10000);

				if(readbytes > (long)widthOrg*heightOrg*30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*30+10000);
					exit(1);
				}
			}
			zip_fclose(fle);
			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}

	MinimalImageB* getImageRaw3_internal(int id, int unused, MinimalImageB*& rimg, MinimalImageB*& gimg, MinimalImageB*& bimg)
	{
	    assert(!use16Bit);
		if(!isZipped)
		{
			return IOWrap::readImageRGB_8U_split(files[id], rimg, gimg, bimg);
		}
		else
		{
			printf("ERROR: Colour currently does not support .zip archive\n");
			exit(1);
		}
	}

	/**
	 * @brief Reads image with exposure
	 * 
	 * Uses the provided 16 or b bit image reader from the IOWrapper
	 * Also gives the exposure and timestamp of the image to the ImageAndExposure object
	 * 
	 * @param id 
	 * @param unused 
	 * @return ImageAndExposure* 
	 */
	ImageAndExposure* getImage_internal(int id, int unused)
	{
	    if(use16Bit)
        {
			if(useColour){
				printf("ERROR: Colour currently does not support 16-bit\n");
				exit(1);
			} else {
				MinimalImage<unsigned short>* minimg = IOWrap::readImageBW_16U(files[id]);
				assert(minimg);
				ImageAndExposure* ret2 = undistort->undistort<unsigned short>(
						minimg,
						(exposures.size() == 0 ? 1.0f : exposures[id]),
						(timestamps.size() == 0 ? 0.0 : timestamps[id]),
						1.0f / 256.0f);
				delete minimg;
				return ret2;
			}
        }
		else
        {
			if(useColour){
				MinimalImageB* rimg; MinimalImageB* gimg; MinimalImageB* bimg;
				MinimalImageB* minimg = getImageRaw3_internal(id, 0, rimg, gimg, bimg);

				ImageAndExposure* ret2 = undistort->undistort<unsigned char>(
						minimg,
						(exposures.size() == 0 ? 1.0f : exposures[id]),
						(timestamps.size() == 0 ? 0.0 : timestamps[id]),
						1, true);

				undistort->undistort_colour<unsigned char>(
					rimg, gimg, bimg,
					ret2,
					(exposures.size() == 0 ? 1.0f : exposures[id]),
					(timestamps.size() == 0 ? 0.0 : timestamps[id])
				);
				
				delete minimg;
				delete rimg; delete gimg; delete bimg;
				return ret2;
			} else {
		MinimalImageB* minimg = getImageRaw_internal(id, 0);
		ImageAndExposure* ret2 = undistort->undistort<unsigned char>(
				minimg,
				(exposures.size() == 0 ? 1.0f : exposures[id]),
				(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		delete minimg;
		return ret2;
	}
        }
	}

	/**
	 * @brief Loads the timestamps
	 * 
	 * Will also load exposures if provided
	 * 
	 */
	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0,path.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			long long id;
			double stamp;
			float exposure = 0;

			if(3 == sscanf(buf, "%lld %lf %f", &id, &stamp, &exposure))
			{
                ids.push_back(id);
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(2 == sscanf(buf, "%lld %lf", &id, &stamp))
			{
                ids.push_back(id);
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// Check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size()==(int)getNumImages()) ;
		for(int i=0;i<(int)exposures.size();i++)
		{
			if(exposures[i] == 0)
			{
				// fix!
				float sum=0,num=0;
				if(i>0 && exposures[i-1] > 0) {sum += exposures[i-1]; num++;}
				if(i+1<(int)exposures.size() && exposures[i+1] > 0) {sum += exposures[i+1]; num++;}

				if(num>0)
					exposures[i] = sum/num;
			}

			if(exposures[i] == 0) exposuresGood=false;
		}


		if((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}



	std::vector<long long> ids; // Saves the ids that are used by e.g. the EuRoC dataset.
	std::vector<ImageAndExposure*> preloadedImages;
	std::vector<std::string> files;
	std::vector<double> timestamps;
	std::vector<float> exposures;

	int width, height;
	int widthOrg, heightOrg;

	std::string path;
	std::string calibfile;

	bool isZipped;
	bool use16Bit;
	bool useColour;

#if HAS_ZIPLIB
	zip_t* ziparchive;
	char* databuffer;
#endif
};

