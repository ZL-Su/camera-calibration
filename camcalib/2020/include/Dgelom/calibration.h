#pragma once

#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <opencv2/core.hpp>
#include "io_manager.h"

namespace zl {
	namespace calib {
		///<brief> camera structural types </brief>
			///<params> [single] - single camera; \
				        [stereo] - binocular cameras </params> 
		enum Type { single = 0, stereo = 1 };
		namespace detail {

			///<brief> camera calibration base class </brief>
			template<typename _Ty> class Base_
			{
			public:
				using value_t = _Ty;
				using pointer = value_t*;
				using point2_t = cv::Vec<value_t, 2>;// Vec2_<value_t>;
				using point3_t = cv::Vec<value_t, 3>;//Vec3_<value_t>;
				using vec3_t = point3_t;
				using matrix_t = cv::Mat_<value_t>;//types::mat_<value_t>;
				using vmatrx_t = std::vector<matrix_t>;
				using image_t = cv::Mat;//types::matuc_t;

				typedef struct Pattern
				{
					enum Type { grid, circ };
					enum Unit { mm, cm, m, inch };
					Pattern() {}
					Pattern(int rows, int cols, float size, Type type = grid, Unit unit = mm)
						:m_type(type), m_unit(unit),
						m_rows(rows), m_cols(cols), m_size(size) {}
					Pattern(const Pattern& _other)
						:m_type(_other.m_type), m_unit(_other.m_unit),
						m_rows(_other.m_rows), m_cols(_other.m_cols), m_size(_other.m_size) {}

					Type  m_type;
					int   m_rows, m_cols;
					float m_size;
					Unit  m_unit;

					bool empty() const noexcept { 
						return m_rows == 0 || m_cols == 0 || m_size == 0; 
					}

					template<typename _Size>
					_Size size() const noexcept { 
						return _Size(m_cols, m_rows); 
					}

					///<brief> grid-corner or circle-center numbers </brief>
					//READONLY_PROPERTY(int, count);
					//__get__(count) { return (m_rows * m_cols); };

					CONST_PROPERTY(int, count, m_rows*m_cols);

					///<brief> SI unit used on current calibration board </brief>
					READONLY_PROPERTY(float, unit);
					__get__(unit) {
						switch (m_unit) {
						case mm: return 1.0f; case cm: return 10.0f; case  m: return 1000.0f;
						case inch: return 25.4f; default: return 1.0f;
						}
					};

					///<brief></brief>
					READONLY_PROPERTY(bool, square);
					__get__(square) { return (m_rows == m_cols); }

				} pattern_t;

				Base_() 
					: m_normal(3, 3) {
				};
				Base_(const pattern_t& _pattn) 
					: m_pattern(_pattn),
					m_normal(3, 3) {
				};

				inline const vmatrx_t& get_impoints() const noexcept {
					return m_vimpts;
				}

				inline const matrix_t& get_obpoints() const noexcept {
					return m_obpts; 
				}

				inline const vmatrx_t& get_vparams() const noexcept {
					return m_vparams; 
				}

				inline value_t error() const noexcept {
					return m_reprojerr[1];
				}

				inline auto image_size() const noexcept {
					return std::make_tuple(m_iw, m_ih); 
				}

				template<typename Functor> 
				inline void execute(Functor fn) noexcept {
					fn; 
				}

			protected:
				template<typename _T1, typename _T2>
				vec3_t _Reproject_to_pcs(vec3_t & pt, const _T1 * pR, const _T2 * pT) const;

			protected: /*protected fields*/
				pattern_t m_pattern;

				/*structure of array for object points,
				  points are stored in adjacent three rows:
				  row[0] = {x0, x1, ..., xn},
				  row[1] = {y0, y1, ..., yn},
				  row[2] = {z0, z1, ..., zn}.*/
				matrix_t m_obpts;

				/*structure of array for image points, same as m_obpts*/
				vmatrx_t m_vimpts;

				value_t m_epsilon = std::numeric_limits<value_t>::epsilon();

				int m_elesize = sizeof(value_t) * 8;
				
				/*The 1st row holds the five intrinsic parameters in order as [fx, s, cx ,fy, cy];
				  distortion factors are stored in the last row, the rest rows hold the external parameters for each pose.*/
				vmatrx_t m_vparams;

				//fx, fy, cx, cy, k1, k2, fs
				std::array<value_t, 7> m_current_iparams;

				matrix_t m_normal;

				value_t m_reprojerr[2]{ 0, 0 };

				size_t m_iw, m_ih;

				size_t m_nthr{ 1 };
			};

			/*
				<brief> Generic class: calibration  algorithms </brief>
				<note> This class is implemented with non-thread safe methods, if it is used in multithreaded environment, each thread should has an instance separately. </note>
			*/
			template<typename _Ty> 
			class Impl_ : public Base_<_Ty>
			{
				using _Myt = Impl_<_Ty>;
				using _Mybase = Base_<_Ty>;

				using typename _Mybase::point2_t;
				using typename _Mybase::point3_t;
				using typename _Mybase::pointer;
				using typename _Mybase::vec3_t;
				using typename _Mybase::image_t;

			public:
				using value_t = typename _Mybase::value_t;
				using matrix_t = typename _Mybase::matrix_t;
				using vmatrx_t = typename _Mybase::vmatrx_t;
				using typename _Mybase::pattern_t;

				Impl_() = default;
				Impl_(const pattern_t& _pattn) : _Mybase(_pattn) {}

				//\brief: the detected patterns are stored in m_vimpts
				//\param: Dir has a structure as {std::string m_fpath; shared_ptr<std::vector<std::string>> m_fnames;}
				//template<typename Dir> vmatrx_t pattern_detect(Dir& _fdir);
				template<typename Dir> 
				void pattern_detect(Dir& _fdir);

				//\brief: load user supplied corner points
				void load_image_points(const vmatrx_t& pts);

				//\brief: load user supplied corner points with new image size {w, h}
				void load_image_points(const vmatrx_t& pts, size_t w, size_t h);

				void set_num_thread(size_t nthr = 1) noexcept;

				//\brief: calibration driver
				//\return: compact calibration parameter, see "Info" in returned matrix
				matrix_t calibrate();

				std::string report() const noexcept;
#ifdef ZL_SIMUL_MODEL
				template<typename Dir> 
				vmatrx_t& load_simulate_model(Dir& _fdir)
				{
					const std::string& path = _fdir.fpath;
					const auto& fnames = _fdir.fnames;
					FILE* fpm = fopen((path + (*fnames)[0]).c_str(), "rt");
					FILE* fpi1 = fopen((path + (*fnames)[1]).c_str(), "rt");
					FILE* fpi2 = fopen((path + (*fnames)[2]).c_str(), "rt");
					FILE* fpi3 = fopen((path + (*fnames)[3]).c_str(), "rt");
					FILE* fpi4 = fopen((path + (*fnames)[4]).c_str(), "rt");
					FILE* fpi5 = fopen((path + (*fnames)[5]).c_str(), "rt");

					if (fpi1 == nullptr || fpi2 == nullptr ||
						fpi3 == nullptr || fpi4 == nullptr ||
						fpi5 == nullptr || fpm == nullptr)
					{
						printf("Arq error\n");
					}

					using Point2d = Dgelo::types::Vec_<double, 2>;
					Point2d model_point, image_point;
					std::vector<Point2d> model_points;
					for (int n = 0; !feof(fpm); ++n)
					{
						fscanf(fpm, "%lf %lf ", &model_point.x, &model_point.y);
						model_points.push_back(model_point);
					}
					fclose(fpm);
					m_obpts.create(2, model_points.size());
					int i = 0;
					for (const auto& point : model_points)
					{
						m_obpts[0][i] = point.x, m_obpts[1][i] = point.y;
						i++;
					}
					m_vimpts.resize(5);
					for (auto& m : m_vimpts) m = matrix_t(4, m_obpts.cols);
					for (i = 0; i < m_obpts.cols; ++i)
					{
						auto& impts1 = m_vimpts[0];
						fscanf(fpi1, "%lf %lf ", &impts1[0][i], &impts1[1][i]);
						auto& impts2 = m_vimpts[1];
						fscanf(fpi2, "%lf %lf ", &impts2[0][i], &impts2[1][i]);
						auto& impts3 = m_vimpts[2];
						fscanf(fpi3, "%lf %lf ", &impts3[0][i], &impts3[1][i]);
						auto& impts4 = m_vimpts[3];
						fscanf(fpi4, "%lf %lf ", &impts4[0][i], &impts4[1][i]);
						auto& impts5 = m_vimpts[4];
						fscanf(fpi5, "%lf %lf ", &impts5[0][i], &impts5[1][i]);
					}
					fclose(fpi1); fclose(fpi2); fclose(fpi3);
					fclose(fpi4); fclose(fpi5);

					m_normal = { value_t(2) / 640, 0, -1, 0, value_t(2) / 480, -1, 0, 0, 1 };
					for (auto& imp : m_vimpts)
					{
						for (int i = 0; i < imp.cols; ++i)
						{
							imp[2][i] = imp[0][i] * m_normal(0) - 1.0;
							imp[3][i] = imp[1][i] * m_normal(4) - 1.0;
						}
					}
					return (m_vimpts);
				}
#endif
				inline const vmatrx_t& get_vparams() const noexcept = delete;
			private:
				matrix_t _Analysis(matrix_t& params);
				matrix_t _Optimize(matrix_t& params);
				matrix_t& _Get_planar_obpts();
				matrix_t  _Get_homography_oi(const matrix_t& _impts);

#ifdef ZL_SIMUL_MODEL
				matrix_t& _Get_simu_model_points() { return m_obpts; }
#endif
				std::string _Mymessage;
				std::string _Myreport;
				matrix_t _Myparams;
			protected:
				using _Mybase::m_obpts;
				using _Mybase::m_vimpts;
				using _Mybase::m_epsilon;
				using _Mybase::m_pattern;
				using _Mybase::m_normal;
				using _Mybase::m_reprojerr;
			};
		};

		///<brief> vision calibration driver </brief>
		template<typename _Ty, Type _Tp, size_t _N = _Tp+1>
		class Calibrator_ final : public detail::Impl_<_Ty>
		{
			using _Mybase = detail::Impl_<_Ty>;
		public:
			using fdir_t = io::Dir_<std::conditional_t<_Tp == Type::single, types::vvstr_t, types::vvstr_t>>;
			using typename _Mybase::matrix_t;
			using typename _Mybase::value_t;
			using typename _Mybase::pattern_t;

			Calibrator_(fdir_t&& _fdirnv, const pattern_t& _pattn)
				: m_fdirnv(std::move(_fdirnv)), _Mybase(_pattn) {
				m_vparams.resize(_N);
			}

			Calibrator_(const pattern_t& _pattn)
				:_Mybase(_pattn) {
				m_vparams.resize(_N);
			}

			//thread-safe engine to invoke calibrator for the _CamIdx-th camera
			template<size_t _CamIdx = 0> 
			matrix_t run()
			{
				const auto& fnamesnv = *m_fdirnv.m_fnames.get();

				if (fnamesnv.empty()) return {};
				if (fnamesnv[_CamIdx].empty()) return {};

				io::Dir_<types::vstr_t> fdir(m_fdirnv.m_fpath, fnamesnv[_CamIdx]);
				_Mybase::pattern_detect(fdir);

				try {
					auto pars = _Mybase::calibrate();
					
					return m_vparams[_CamIdx] = std::move(pars);
				}
				catch (std::exception& e) {
					std::cout << e.what() << std::endl;
				}
			};

			//dimension of each matrix should be 4-rows x N-cols, where N is the point count of each view.
			//data format: 1st and 2nd rows are the x- and y-componenst of all detected points respectively, the rest 2 rows are reserved work space. 
			void set_image_points(const std::vector<matrix_t>& vimpts) {
				_Mybase::load_image_points(vimpts);
			}
			void set_image_points(const std::vector<matrix_t>& vimpts, size_t image_width, size_t image_height) {
				_Mybase::load_image_points(vimpts, image_width, image_height);
			}

			//width, height: image width/height in projector
			//note: set image points with method "set_image_points()" before calling this method.
			matrix_t run()
			{
				return std::move(_Mybase::calibrate());
			}

		private:
			//include (1) path, (2) image name list for all views, in which views are grouped by using outer vector. 
			//The inner vector is used to store image names for each outer view vector 
			fdir_t m_fdirnv;
			using _Mybase::m_vparams;
			using _Mybase::m_vimpts;
			using _Mybase::m_pattern;
			using _Mybase::m_normal;
		};

	};

	using calibration_type = calib::Calibrator_<double_t, calib::Type::stereo>;
};

//@param: _pintrinsic = {fx, fy, cx, cy}
template<typename PointIn, typename PointOut, typename _Ty>
inline PointOut cvt_pixcoord_to_imgcoord(const PointIn& _pix_pt, const _Ty _pintrinsic[4])
{
	const auto fx = _pintrinsic[0], fy = _pintrinsic[1];
	const auto cx = _pintrinsic[2], cy = _pintrinsic[3];
	auto x = (_pix_pt.x - cx) / fx, y = (_pix_pt.y - cy) / fy;
	return PointOut(x, y, 1);
}

template<typename PointIn, typename PointOut, typename _Ty>
inline PointOut cvt_imgcoord_to_camcoord(const PointIn& _img_pt, const _Ty& _f)
{
	return Point(_img_pt.x, _img_pt.y, _f);
}