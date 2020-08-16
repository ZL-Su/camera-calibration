#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "Dgelom\io_manager.h"
#include "Dgelom\calibration.h"

void example()
{
	using calibration_t = zl::calibration_type;

	// Push calibration image names into a vector
	std::vector<std::string> fnames = //image name list
	{
		/*"DJI_00019_0.bmp", "DJI_00021_0.bmp", "DJI_00023_0.bmp",
		"DJI_00025_0.bmp", "DJI_00026_0.bmp", "DJI_00027_0.bmp",
		"DJI_00028_0.bmp", "DJI_00029_0.bmp", "DJI_00032_0.bmp",*/
		"0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", 
		"6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg",
		"12.jpg", "13.jpg", "14.jpg", "15.jpg",
	};
	
	// Make calibration image directory
	calibration_t::fdir_t fdirnv(
		//"E:/SciR/Datum/calib_imgs/Olympus-50mm/", //Folder path where the calibration images are stored.
		"E:/SciR/Datum/calib_imgs/7x7/", //Folder path where the calibration images are stored.
		{ fnames } //Calibration image name list.
	);

	// Create calibration instante, requires:
	calibration_t cam_calib(
		std::move(fdirnv),                  // calibration image directory info.
		{                                   // planar pattern info with
			7, 7, 20,                      //   8-rows, 11-cols, spacing 20~mm,
			calibration_t::pattern_t::grid  //   and circular corner[or grid for chessboard]
		}
	);

	// Set the number of threads
	cam_calib.set_num_thread(8);
	// Run clibrator
	const auto cam_params = cam_calib.run<>();

	// Print reprojection error
	std::cout << " >>>Reprojection error of camera calibration:" << cam_calib.error() << "\n";

	// Print internal parameters
	std::cout << " >>>Internal paramters: "
		<< "\n    fx, fy = " << cam_params[0][0] << ", " << cam_params[0][3]
		<< "\n    cx, cy = " << cam_params[0][2] << ", " << cam_params[0][4]
		<< "\n    k1, k2 = " << cam_params[cam_params.rows - 1][0] << ", " << cam_params[cam_params.rows - 1][1]
		<< "\n    p1, p2 = " << cam_params[cam_params.rows - 1][2] << ", " << cam_params[cam_params.rows - 1][3]
		<< std::endl;

	// Print report
	std::cout << cam_calib.report();
	
	//-----------------I-AM-THE-LOVELY-PARTING-LINE------------------------

	// Get detected corners or centers
	auto impts = cam_calib.get_impoints();
	// Get calibration image size
	const auto[w, h] = cam_calib.image_size();

	//-----------------I-AM-THE-LOVELY-PARTING-LINE------------------------

	// Create projector calibrator
	calibration_t pro_calib(calibration_t::pattern_t{ 11, 8, 20 });

	// Set corner or center points in projector and image size [w, h]
	pro_calib.set_image_points(impts, w, h);

	// Set the number of threads
	pro_calib.set_num_thread(8);

	// Run clibrator and return calibration parameters
	auto pro_params = pro_calib.run();

	// Print reprojection error
	std::cout << " Reprojection error of projector calibration:" << pro_calib.error() << "\n";

	// Print internal parameters
	std::cout << " >>>Internal paramters: "
		<< "\n    fx, fy = " << pro_params[0][0] << ", " << pro_params[0][3]
		<< "\n    cx, cy = " << pro_params[0][2] << ", " << pro_params[0][4]
		<< "\n    k1, k2 = " << pro_params[pro_params.rows - 1][0] << ", " << pro_params[pro_params.rows - 1][1]
		<< "\n    p1, p2 = " << pro_params[pro_params.rows - 1][2] << ", " << pro_params[pro_params.rows - 1][3]
		<< std::endl;

	// Print report
	std::cout << pro_calib.report();
}
