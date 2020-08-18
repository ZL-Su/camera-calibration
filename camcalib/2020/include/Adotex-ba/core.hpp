/*************************************************************************
  Dgelom Adotex-ba, a fast bundle adjustment enhanced by Adotex.
  Copyright(C) 2020, Dgelom Su, all rights reserved.
**************************************************************************/
#pragma once
#include "port.hpp"
#include <array>
#include <vector>
#include <string>
#include <type_traits>

DGELOM_NAMESPACE_BEGIN

enum SubsystemType {
	CONSOLE,
	WINDOWS
};

class ADOTEX_BA_API AdotexMLE {
public:
	using value_t = double_t;
	using pointer = std::add_pointer_t<value_t>;
	using point2d_t = std::array<value_t, 2>;
	using point3d_t = std::array<value_t, 3>;
	using index_t = typename point2d_t::difference_type;

	/**
	 * \brief Holds the internal paramters of a camera to be calibrated.
	 * \field data Internal parameters with {fx, fs, cx, fy, cy, k1, k2, p1, p2}.
	 */
	struct internal_param_type
	{
		enum { NUMBER = 9 };
		value_t data[NUMBER];
	};
	using internal_param_t = internal_param_type;

	/**
	 * \brief Holds the external paramters between the camera and object.
	 * \field data External parameters with {rx, ry, rz, tx, ty, tz}.
	 */
	struct external_param_type
	{
		enum { NUMBER = 6 };
		value_t data[NUMBER];
	};
	using external_param_t = external_param_type;

	/**
	 * \brief Options to control the MLE solver.
	 */
	struct options_type
	{
		// Radial distortion mode.
		// Set to "forward" to use U2D model, and to "inverse" to use D2U model.
		// \sa Paper http://dx.doi.org/10.1016/j.optlaseng.2019.03.018 in detail.
		const char rd_mode[8] = "inverse";

		size_t num_iterations = 100;

		size_t num_threads = 1;

		bool fixed_world_frames = false;
	};
	using options_t = options_type;

	AdotexMLE() = delete;

	AdotexMLE(const std::vector<point3d_t>& object_points, 
		const std::vector<std::vector<point2d_t>>& image_points);

	std::string solve(internal_param_t& ipars, 
		std::vector<external_param_t>& epars, 
		options_t options = options_t{});

	value_t total_reprojection_error() const noexcept;
	value_t average_reprojection_error() const noexcept;

	std::string report() const noexcept;

private:
	value_t _Myerror;
	std::vector<std::vector<point3d_t>> _Myobject_points;
	std::vector<std::vector<point2d_t>> _Myimage_points;

	std::string _Myreport = "Adotex is suspended.";
};

class ADOTEX_BA_API AdotexBA {
public:
	using value_t = double_t;
	using pointer = std::add_pointer_t<value_t>;
	using point2d_t = std::array<value_t, 2>;
	using index_t = typename point2d_t::difference_type;
	using array_1x6_t = std::array<value_t, 6>;

	/**
	 * \brief An simple 3D point type
	 */
	struct point3d_type {

		decltype(auto) x() const noexcept { return data[0];}
		decltype(auto) x() noexcept { return data[0];}
		decltype(auto) y() const noexcept { return data[1];}
		decltype(auto) y() noexcept { return data[1];}
		decltype(auto) z() const noexcept { return data[2];}
		decltype(auto) z() noexcept { return data[2];}

		// simd and cache friendly
		value_t data[3];
	};
	using point3d_t = point3d_type;

	/**
	 * \brief Holds the observed correspondence across the left and right image.
	 * \field left The part of the observed correspondence in left image.
	 * \field right The part of the observed correspondence in right image.
	 */
	struct correspondence_type {
		point2d_t left;
		point2d_t right;
	}; 
	using correspondence_t = correspondence_type;

	/**
	 * \brief Stores the camera calibration parameters.
	 * \field left Internal parameters of the left camera.
	 * \field right Internal parameters of the right camera.
	 */
	struct camera_type
	{
		mutable array_1x6_t left;
		mutable array_1x6_t right;

		bool fixed_principal_point = false;
	};
	using camera_t = camera_type;

	/**
	 * \brief Loss function to metric the cost of the residual terms.
	 */
	enum loss_metric_type
	{
		HUBER,
		SOLFL1,
		CAUCH,
		NONE
	};

	/**
	 * \brief Options to control the Adotex BA solver.
	 * \field subsystem Set to CONSOLE for console app and to WINDOWS for GUI app.
	 * \field loss_argument Set argument to control the robustification.
	 * \field is_camera_fixed Set true or false to make camera calibration varying.  
	 */
	struct options_type
	{
		SubsystemType subsystem = CONSOLE;
		value_t loss_argument = 1.;
		loss_metric_type loss_metric = CAUCH;
		bool is_camera_fixed = false;
	};
	using options_t = options_type;

	AdotexBA() = delete;

	/**
	 * \brief Create Adotex-ba solver with a set of stereo correspondences. 
	 * \param point_correspondences Input observed stereo correspondences, 
	                                \sa "correspondence_type".
	 * \param options Congf adotex solver properties. \sa options_type 
	 */
	AdotexBA(std::vector<correspondence_t>&& point_correspondences, 
		options_type options);

	virtual ~AdotexBA();

	/**
	 * \brief Initialize the Adotex-ba solver with initial paramters.
	 * \param cameras Initial internal calibration paramters for both cameras,
	                  \sa "camera_type".
	 * \param geometry Initial vision geometry including relative rotation and translation,
	                  \st geometry = {rx, ry, rz, tx, ty, tx}.
	 */
	void init(const camera_t& cameras, const array_1x6_t& geometry) noexcept;
	/**
	 * \brief Initialize the Adotex-ba solver with initial paramters.
	 * \param cameras Initial internal calibration paramters for both cameras,
					  \sa "camera_type".
	 * \param geometry Initial vision geometry including relative rotation and translation,
					  \st geometry = {rx, ry, rz, tx, ty, tx}.
	 * \param depths Initial depth values corresponds to each of stereo correspondences.
	 */
	void init(const camera_t& cameras, const array_1x6_t& geometry, const std::vector<value_t>& depths);

	/**
	 * \brief Solve a bundle adjustment model with Adotex solver.
	 * \params max_iters Max number of iterations to be used by Adotex.
	 * \params nthrs Number of threads to launch Adotex solver.
	 * \return Reason why Adotex solver termination.
	 */
	std::string solve(size_t max_iters = 50, size_t nthrs = 4);

	// Following properties below to get results, \sa each method name...

	value_t reprojection_error() const noexcept;

	camera_t get_camera_parameters() const noexcept;

	array_1x6_t get_vision_geometry() const noexcept;

	std::vector<point3d_t> get_structure() const noexcept;

private:
	value_t _Myerror;

	options_type _Myopts;

	camera_type _Mycams;

	array_1x6_t _Mygeom;
	
	std::vector<value_t> _Myinvd;

	std::vector<correspondence_t> _Mydata;

	std::string _Mymsg = "Adotex-ba";

	std::string _Myreport = "Adotex is suspended.";
};

DGELOM_NAMESPACE_END
