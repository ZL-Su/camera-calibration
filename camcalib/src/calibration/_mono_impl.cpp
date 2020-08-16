#include "mono_calibrator.h"
#include "cost_functions.hpp"
#include <Matrice/core/vector.h>

namespace dgelom {
template<typename T, pattern_type _Patt, distortion_model _Model>
void mono_calibrator<T, _Patt, _Model>::_Optimize()
{
	auto _Nviews = this->m_effindices.size();
	auto _Nctrls = this->m_points.cols();

	auto &_Fx = m_params(0), &_Fy = m_params(1);
	auto &_Cx = m_params(2), &_Cy = m_params(3);
	auto &_K1 = m_params(4), &_K2 = m_params(5);
	auto &_Fs = m_params(6);

	Matrix<default_type> _Plane_points(_Nctrls, 3);
	transform(m_points[0], m_points[0] + _Nctrls, _Plane_points.begin(), 3);
	transform(m_points[1], m_points[1] + _Nctrls, _Plane_points.begin() + 1, 3);
	fill(_Plane_points.begin() + 2, _Plane_points.end(), 3, default_type(0));

	Matrix_<default_type, 7, 1> _Vars = 1;
	_Vars(4) = _Vars(5) = 0.001;
	auto _Problem = std::make_shared<xop::Problem>();
	for (auto _Idx = 0; _Idx < _Nviews; ++_Idx) {
		auto _Row = m_effindices[_Idx] << 1;
		auto _Pose = m_poses.ptr(_Idx);
		for (index_t i = 0; i < _Nctrls; ++i) {
			const auto _U0 = m_imgpts[_Row][i];
			const auto _V0 = m_imgpts[_Row+1][i];
			const Vec_<default_type, 2> _Point{ _U0, _V0 };
			auto _Cost = CostFunc::_Std_mono_error<decltype(_Point), Dmodel>
				::Create(_Fx, _Fy, _Cx, _Cy, _Point);
			_Problem->AddResidualBlock(_Cost, new xop::HuberLoss(0.05), _Vars.begin(), _Pose, _Plane_points[i]);
		}
	}

	xop::Solver::Options _Options;
	_Options.linear_solver_type = xop::DENSE_SCHUR;
	_Options.max_num_iterations = 500;
	_Options.num_threads = 8;
#ifdef DGELOM_CONSOL_DEBUG
	_Options.minimizer_progress_to_stdout = true;
#endif

	auto _Summary = std::make_unique<xop::Solver::Summary>();
	xop::Solve(_Options, _Problem.get(), _Summary.get());
#ifdef DGELOM_CONSOL_DEBUG
	std::cout << _Summary->FullReport() << std::endl;
#endif
	m_params = m_params * _Vars;

	m_error = _Reprojection_error(_Problem);
}

template void mono_calibrator<double, pattern_type::squared, distortion_model::D2U>::_Optimize();
template void mono_calibrator<double, pattern_type::circular, distortion_model::D2U>::_Optimize();
template void mono_calibrator<double, pattern_type::squared, distortion_model::U2D>::_Optimize();
template void mono_calibrator<double, pattern_type::circular, distortion_model::U2D>::_Optimize();
}